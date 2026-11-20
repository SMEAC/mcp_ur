"""A simulated UR e-Series arm (UR3e or UR5e) that runs the scripts this server generates.

It parses ``movel`` / ``movej`` / ``stopl`` lines from the URScript text and
plays them back in real time (or faster, via ``speedup``), updating the same
state the RTDE interface reports: TCP pose, speed, runtime state, robot and
safety mode. Like the real controller, a new script interrupts the old one.

Simplifications: ``movej`` to a pose is moved in a straight line, the joint
angles are only updated by ``movej`` with joint targets, and there is no
collision model. ``obstacle_z_m`` simulates a probe touching something: a
force-guarded move that goes below it halts the program there.
"""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, replace

from .geometry import (
    URModel,
    forward_kinematics,
    get_model,
    interpolate_pose,
    orientation_distance,
    position_distance,
    pose_trans,
)
from .transport import ROBOT_MODES, SAFETY_MODES, RobotState

DEFAULT_SIM_MODEL = "ur3e"
SIM_HOME_DEG = (0.0, -90.0, 90.0, -90.0, -90.0, 0.0)
SIM_TCP = [0.0, 0.0, 0.100, 0.0, 0.0, 0.0]  # a 100 mm probe holder on the flange

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_MOVE_RE = re.compile(
    rf"^\s*(movel|movej)\(\s*(p?)\[([^\]]*)\]\s*(?:,\s*a\s*=\s*({_NUM}))?\s*(?:,\s*v\s*=\s*({_NUM}))?"
)
_STOP_RE = re.compile(r"^\s*(stopl|stopj)\(")


def is_sim(host: str | None) -> bool:
    """'SIM', 'SIMULATOR', or 'SIM:<model>' such as 'SIM:UR5E'."""
    if host is None:
        return False
    h = host.strip().upper()
    return h in ("SIM", "SIMULATOR") or h.startswith("SIM:")


def sim_model(host: str) -> str | None:
    """The model named in 'SIM:UR5E', or None for plain 'SIM'."""
    h = host.strip()
    if ":" not in h:
        return None
    return h.split(":", 1)[1].strip() or None


@dataclass
class _Move:
    kind: str           # movel | movej_pose | movej_joints | stop
    target: list[float]
    a: float
    v: float
    guarded: bool = False


class SimulatedUR:
    def __init__(self, speedup: float = 1.0, model: URModel | str = DEFAULT_SIM_MODEL):
        self.model = get_model(model) if isinstance(model, str) else model
        self.host = f"SIM:{self.model.name.upper()}"
        self.speedup = speedup
        self._lock = threading.Lock()
        self._q = [math.radians(d) for d in SIM_HOME_DEG]
        self._st = RobotState(robot_mode=7, safety_mode=1, runtime_state=1, status_bits=0b1,
                              q=list(self._q), tcp_pose=self._fk(self._q), received=time.monotonic())
        self._st.target_tcp_pose = list(self._st.tcp_pose)
        self._gen = 0
        self._thread: threading.Thread | None = None
        self.scripts: list[str] = []      # everything received, for tests and debugging
        self.log: list[str] = []          # textmsg() output
        self.obstacle_z_m: float | None = None
        self.remote_control = True

    def _fk(self, q: list[float]) -> list[float]:
        return pose_trans(forward_kinematics(q, self.model), SIM_TCP)

    # ---- RobotBackend ---------------------------------------------------

    def state(self) -> RobotState:
        with self._lock:
            return replace(self._st, received=time.monotonic(), q=list(self._st.q),
                           tcp_pose=list(self._st.tcp_pose), tcp_speed=list(self._st.tcp_speed))

    def send_script(self, script: str) -> None:
        self.scripts.append(script)
        with self._lock:
            if not self.remote_control or self._st.robot_mode != 7 or self._st.safety_mode not in (1, 2):
                return  # the real controller ignores (or rejects) the program in these states
            moves = self._parse(script)
            self._gen += 1
            gen = self._gen
            self._st.runtime_state, self._st.status_bits = 2, 0b11
        self._thread = threading.Thread(target=self._play, args=(gen, moves), daemon=True)
        self._thread.start()

    def dashboard(self, command: str) -> str:
        c = command.strip().lower()
        with self._lock:
            st = self._st
            if c == "robotmode":
                return f"Robotmode: {ROBOT_MODES[st.robot_mode]}"
            if c == "safetystatus" or c == "safetymode":
                return f"Safetystatus: {SAFETY_MODES[st.safety_mode]}"
            if c == "is in remote control":
                return "true" if self.remote_control else "false"
            if c == "polyscopeversion":
                return "URSoftware 5.15.0 (simulated)"
            if c == "get robot model":
                return self.model.dashboard_name
            if c == "get serial number":
                return "SIM0000000"
            if c == "programstate":
                return "PLAYING mcp" if st.runtime_state == 2 else "STOPPED"
            if c == "running":
                return f"Program running: {str(st.runtime_state == 2).lower()}"
            if c == "power on":
                if st.robot_mode in (3, 5, 7):
                    st.robot_mode = 5 if st.robot_mode == 3 else st.robot_mode
                return "Powering on"
            if c == "brake release":
                if st.robot_mode in (5, 7):
                    st.robot_mode = 7
                return "Brake releasing"
            if c == "power off":
                self._gen += 1
                st.robot_mode, st.runtime_state, st.status_bits = 3, 1, 0
                return "Powering off"
            if c == "unlock protective stop":
                if st.safety_mode == 3:
                    st.safety_mode = 1
                    return "Protective stop releasing"
                return "Cannot unlock protective stop until 5s after occurrence. Always inspect cause of protective stop before unlocking"
            if c in ("close safety popup", "close popup"):
                return "closing safety popup" if "safety" in c else "closing popup"
            if c in ("stop", "pause"):
                self._gen += 1
                st.runtime_state, st.status_bits = 1, st.status_bits & ~0b10
                return "Stopped" if c == "stop" else "Pausing program"
        return f"could not understand: '{command}'"

    def close(self) -> None:
        with self._lock:
            self._gen += 1

    # ---- fault injection for tests -------------------------------------

    def protective_stop(self) -> None:
        with self._lock:
            self._gen += 1
            self._st.safety_mode, self._st.runtime_state, self._st.status_bits = 3, 1, 0b1
            self._st.tcp_speed = [0.0] * 6

    # ---- internals ------------------------------------------------------

    def _parse(self, script: str) -> list[_Move]:
        moves: list[_Move] = []
        guard_next = False
        for line in script.splitlines():
            if "textmsg(" in line:
                m = re.search(r'textmsg\("([^"]*)"', line)
                if m:
                    self.log.append(m.group(1))
                continue
            if re.match(r"^\s*\w+\s*=\s*run\s+mcp_force_guard", line):
                guard_next = True
                continue
            if _STOP_RE.match(line):
                moves.append(_Move("stop", [], 1.0, 0.0))
                continue
            m = _MOVE_RE.match(line)
            if not m:
                continue
            fn, is_pose, nums, a, v = m.groups()
            target = [float(x) for x in nums.split(",")]
            if len(target) != 6:
                raise ValueError(f"simulator: expected 6 values in {line.strip()!r}")
            kind = "movel" if fn == "movel" else ("movej_pose" if is_pose else "movej_joints")
            if fn == "movel" and not is_pose:
                raise ValueError("simulator: movel with joint targets is not supported")
            dflt_a, dflt_v = (1.2, 0.25) if fn == "movel" else (1.4, 1.05)
            moves.append(_Move(kind, target, float(a or dflt_a), float(v or dflt_v), guard_next))
            guard_next = False
        return moves

    def _duration(self, mv: _Move, start_pose: list[float], start_q: list[float]) -> float:
        if mv.kind == "movej_joints":
            dist = max(abs(a - b) for a, b in zip(mv.target, start_q))
        elif mv.kind == "movej_pose":
            dist = max(position_distance(start_pose, mv.target) / 0.25,
                       orientation_distance(start_pose, mv.target))
        else:
            dist = position_distance(start_pose, mv.target)
        if dist < 1e-9:
            return 0.0
        return dist / mv.v + mv.v / mv.a

    def _play(self, gen: int, moves: list[_Move]) -> None:
        dt = 0.004
        try:
            for mv in moves:
                if mv.kind == "stop":
                    break
                with self._lock:
                    if gen != self._gen:
                        return
                    p0, q0 = list(self._st.tcp_pose), list(self._st.q)
                    self._st.target_tcp_pose = (self._fk(mv.target) if mv.kind == "movej_joints"
                                                else list(mv.target))
                T = self._duration(mv, p0, q0)
                t = 0.0
                prev = p0
                while True:
                    s = 1.0 if T <= 0 else min(1.0, t / T)
                    s_smooth = s * s * (3 - 2 * s)  # smooth start and stop
                    if mv.kind == "movej_joints":
                        q = [a + (b - a) * s_smooth for a, b in zip(q0, mv.target)]
                        pose = self._fk(q)
                    else:
                        q, pose = q0, interpolate_pose(p0, mv.target, s_smooth)
                    halted = (mv.guarded and self.obstacle_z_m is not None and pose[2] < self.obstacle_z_m)
                    with self._lock:
                        if gen != self._gen:
                            return
                        vel = [(a - b) / (dt * self.speedup) for a, b in zip(pose[:3], prev[:3])]
                        self._st.q, self._st.tcp_pose = list(q), (prev if halted else pose)
                        self._st.tcp_speed = [0.0] * 6 if halted or s >= 1.0 else [*vel, 0.0, 0.0, 0.0]
                        self._st.timestamp += dt * self.speedup
                    if halted:
                        self.log.append("MCP: contact force above limit - motion halted")
                        return
                    if s >= 1.0:
                        break
                    prev = pose
                    time.sleep(dt)
                    t += dt * self.speedup
        finally:
            with self._lock:
                if gen == self._gen:
                    self._st.runtime_state, self._st.status_bits = 1, 0b1
                    self._st.tcp_speed = [0.0] * 6
