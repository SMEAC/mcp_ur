"""Plan probe moves and run them on the robot.

Every move to a test point is the same safe sequence:

1. **retract**  - if the tip is below the clearance plane, go straight up
   (along the board normal, orientation unchanged) to the clearance plane;
2. **travel**   - to the position above the target, at clearance height,
   with the probe pointing down (``movel`` when already over the board,
   ``movej`` when coming from further away, e.g. from home);
3. **descend**  - down to ``approach_height_mm`` above the pad;
4. **probe**    - slowly down to ``pad Z - contact_depth_mm``, optionally
   guarded by the e-Series force/torque sensor.

The whole sequence is sent as one URScript program. The server then watches
RTDE until the program has finished and checks that the TCP really is where it
should be.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .board import Board, BoardError, TestPoint
from .geometry import Pose, forward_kinematics, orientation_distance, position_distance
from .transport import SAFETY_MODES, RobotBackend, RobotState
from .urscript import Segment, build_program, stop_program

PLANE_TOL_MM = 2.0
ORIENT_TOL_DEG = 2.0
START_TIMEOUT_S = 3.0


class MotionError(RuntimeError):
    pass


@dataclass
class Plan:
    title: str
    segments: list[Segment]
    final_pose: Pose
    final_joints: list[float] | None = None
    notes: list[str] = field(default_factory=list)

    def describe(self) -> list[dict[str, Any]]:
        return [s.describe() for s in self.segments]


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def _mm_s(v: float) -> float:
    return v / 1000.0


def _deg(v: float) -> float:
    return math.radians(v)


def plan_retract(board: Board, pose: Pose) -> list[Segment]:
    """Straight up along the board normal to the clearance plane, if below it."""
    m = board.motion
    z = board.to_local(pose)["z_mm"]
    if z >= m.clearance_height_mm - 0.5:
        return []
    lift = (m.clearance_height_mm - z) / 1000.0
    target = [*(np.asarray(pose[:3]) + board.normal() * lift), *pose[3:]]
    board.check_reachable(target, "retract")
    return [Segment("movel", [float(x) for x in target], _mm_s(m.retract_speed_mm_s),
                    _mm_s(m.travel_accel_mm_s2), f"retract to clearance plane ({m.clearance_height_mm:g} mm)")]


def _travel_kind(board: Board, start: Pose) -> str:
    m = board.motion
    if m.travel_motion != "auto":
        return "movel" if m.travel_motion == "movel" else "movej_pose"
    loc = board.to_local(start)
    over_board = abs(loc["z_mm"] - m.clearance_height_mm) <= PLANE_TOL_MM and loc["probe_tilt_deg"] < 5
    return "movel" if over_board else "movej_pose"


def _travel_segment(board: Board, start: Pose, target: Pose, label: str) -> Segment:
    m = board.motion
    kind = _travel_kind(board, start)
    if kind == "movel":
        return Segment("movel", target, _mm_s(m.travel_speed_mm_s), _mm_s(m.travel_accel_mm_s2), label)
    return Segment("movej_pose", target, _deg(m.joint_speed_deg_s), _deg(m.joint_accel_deg_s2), label)


def plan_to_point(board: Board, pose: Pose, tp: TestPoint, stop_above: bool = False) -> Plan:
    m = board.motion
    segs = plan_retract(board, pose)
    start = segs[-1].target if segs else pose

    above_z = min(tp.z_mm + board.approach_height(tp), m.clearance_height_mm)
    contact_z = board.contact_z(tp)
    over = board.point_pose(tp, m.clearance_height_mm)
    approach = board.point_pose(tp, above_z)
    contact = board.point_pose(tp, contact_z)
    for p, what in ((over, "travel"), (approach, "approach"), (contact, "contact")):
        board.check_reachable(p, f"{tp.id} {what}")

    segs.append(_travel_segment(board, start, over, f"travel above {tp.id} at clearance height"))
    if m.clearance_height_mm - above_z > 0.05:
        segs.append(Segment("movel", approach, _mm_s(m.descend_speed_mm_s), _mm_s(m.travel_accel_mm_s2),
                            f"descend to {board.approach_height(tp):g} mm above {tp.id}"))
    if stop_above:
        return Plan(f"hover above {tp.id}", segs, approach)
    segs.append(Segment("movel", contact, _mm_s(m.probe_speed_mm_s), _mm_s(m.probe_accel_mm_s2),
                        f"probe {tp.id} slowly ({m.probe_speed_mm_s:g} mm/s)", force_limit_n=m.max_force_n))
    return Plan(f"probe {tp.id}", segs, contact)


def plan_home(board: Board | None, pose: Pose, home_deg: list[float], joint_speed_deg_s: float = 45.0,
              joint_accel_deg_s2: float = 90.0) -> Plan:
    segs = plan_retract(board, pose) if board else []
    q = [math.radians(d) for d in home_deg]
    if board:
        joint_speed_deg_s, joint_accel_deg_s2 = board.motion.joint_speed_deg_s, board.motion.joint_accel_deg_s2
    segs.append(Segment("movej_joints", q, _deg(joint_speed_deg_s), _deg(joint_accel_deg_s2), "move to home"))
    model = board.robot if board and board.robot else "ur3e"  # final_pose is informational; joints are checked
    return Plan("home", segs, forward_kinematics(q, model), final_joints=q)


def plan_nudge(board: Board, tp: TestPoint, offset_mm: list[float]) -> Plan:
    m = board.motion
    dx, dy, dz = offset_mm
    if math.sqrt(dx * dx + dy * dy + dz * dz) > m.max_nudge_mm + 1e-9:
        raise BoardError(f"Total correction {offset_mm} mm from the nominal {tp.id} position exceeds "
                         f"max_nudge_mm={m.max_nudge_mm}. If the point is wrong, fix it in the board file "
                         "or re-teach it with robot_teach(operation='record_point').")
    if dz < -1.0:
        raise BoardError("A nudge may go at most 1 mm deeper than the nominal contact depth")
    target = board.local_pose(tp.x_mm + dx, tp.y_mm + dy, board.contact_z(tp) + dz, board.tool_rz(tp))
    board.check_reachable(target, "nudge")
    seg = Segment("movel", target, _mm_s(m.probe_speed_mm_s), _mm_s(m.probe_accel_mm_s2),
                  f"nudge {tp.id} by {[round(v, 3) for v in offset_mm]} mm", force_limit_n=m.max_force_n)
    return Plan(f"nudge {tp.id}", [seg], target)


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def _estimate_s(plan: Plan, start: Pose, start_q: list[float]) -> float:
    total, prev, prev_q = 0.0, start, start_q
    for s in plan.segments:
        if s.kind == "movej_joints":
            total += max(abs(a - b) for a, b in zip(s.target, prev_q)) / s.v + 2 * s.v / s.a
            prev, prev_q = forward_kinematics(s.target), s.target
        elif s.kind == "movej_pose":
            total += 2 * math.pi / s.v + 2 * s.v / s.a  # worst case: half a turn on some joint
            prev = s.target
        else:
            total += position_distance(prev, s.target) / s.v + 2 * s.v / s.a
            prev = s.target
    return total


def at_pose(st: RobotState, pose: Pose, tol_mm: float) -> tuple[bool, float, float]:
    err_mm = position_distance(st.tcp_pose, pose) * 1000
    err_deg = math.degrees(orientation_distance(st.tcp_pose, pose))
    return err_mm <= tol_mm and err_deg <= ORIENT_TOL_DEG, err_mm, err_deg


def arrived(st: RobotState, plan: Plan, tol_mm: float) -> tuple[bool, dict[str, float]]:
    """Joint targets are checked in joint space (the TCP offset doesn't matter), poses in Cartesian space."""
    if plan.final_joints is not None:
        err = math.degrees(max(abs(a - b) for a, b in zip(st.q, plan.final_joints)))
        return err < 0.5, {"joint_error_deg": round(err, 3)}
    ok, err_mm, err_deg = at_pose(st, plan.final_pose, tol_mm)
    return ok, {"position_error_mm": round(err_mm, 3), "orientation_error_deg": round(err_deg, 3)}


def check_ready(robot: RobotBackend, st: RobotState) -> None:
    if not st.can_move:
        d = st.describe()
        hint = {
            "POWER_OFF": "Power on and release the brakes: robot_device(operation='power_on') then 'brake_release'.",
            "IDLE": "Release the brakes: robot_device(operation='brake_release').",
            "POWER_ON": "Release the brakes: robot_device(operation='brake_release').",
        }.get(d["robot_mode"], "")
        if d["safety_mode"] == "PROTECTIVE_STOP":
            hint = ("Find out why the robot stopped, then unlock with "
                    "robot_device(operation='unlock_protective_stop').")
        elif d["safety_mode"] not in ("NORMAL", "REDUCED"):
            hint = "Resolve the safety stop on the robot (e-stop, safeguard, ...) at the pendant."
        raise MotionError(f"Robot not ready to move: {d['robot_mode']}, safety {d['safety_mode']}. {hint}")
    if st.program_running:
        raise MotionError("A program is already running on the robot (a pendant program or an earlier move). "
                          "Wait for it, or stop it with robot_move(operation='stop').")
    try:
        remote = robot.dashboard("is in remote control").strip().lower()
    except Exception:
        remote = ""
    if remote == "false":
        raise MotionError("The robot is in Local mode, so it ignores scripts sent over the network. "
                          "Switch the pendant to Remote Control (profile icon top right -> Remote Control).")


def execute(robot: RobotBackend, plan: Plan, tol_mm: float, abort: threading.Event | None = None,
            timeout_s: float | None = None) -> dict[str, Any]:
    st0 = robot.state()
    check_ready(robot, st0)
    name, script = build_program(plan.segments, plan.title)
    est = _estimate_s(plan, st0.tcp_pose, st0.q)
    timeout = timeout_s or (est * 2.0 / max(st0.speed_fraction, 0.05) + 5.0)

    t0 = time.monotonic()
    robot.send_script(script)
    seen_running = False
    while True:
        time.sleep(0.01)
        st = robot.state()
        elapsed = time.monotonic() - t0
        if st.safety_mode not in (1, 2):
            raise MotionError(f"Robot stopped: safety mode {SAFETY_MODES.get(st.safety_mode, st.safety_mode)} "
                              f"after {elapsed:.1f} s. Inspect the robot and the probe before continuing.")
        if abort is not None and abort.is_set():
            raise MotionError("Move cancelled by robot_move(operation='stop').")
        if st.program_running:
            seen_running = True
        elif seen_running:
            break
        elif elapsed > 0.5 and arrived(st, plan, tol_mm)[0] and st.linear_speed < 1e-3:
            break  # very short program that finished between two samples
        elif elapsed > START_TIMEOUT_S:
            raise MotionError(
                f"The robot did not start program {name} within {START_TIMEOUT_S:.0f} s. Check the pendant "
                "log for a script error, and that the robot is in Remote Control mode.")
        if elapsed > timeout:
            robot.send_script(stop_program())
            raise MotionError(f"Move did not finish within {timeout:.0f} s - stopped the robot.")

    time.sleep(0.05)
    st = robot.state()
    ok, errors = arrived(st, plan, tol_mm)
    result: dict[str, Any] = {
        "program": name,
        "duration_s": round(time.monotonic() - t0, 2),
        "final_tcp_mm": [round(v * 1000, 2) for v in st.tcp_pose[:3]],
        **errors,
    }
    if not ok:
        last = plan.segments[-1]
        err_mm, err_deg = at_pose(st, plan.final_pose, tol_mm)[1:]
        if last.force_limit_n and st.tcp_pose[2] > plan.final_pose[2]:
            raise MotionError(
                f"The probe stopped {err_mm:.2f} mm short of the contact position: the TCP force exceeded "
                f"{last.force_limit_n} N during the slow descent. Something is higher than expected (wrong "
                "board position, component, or test point Z). Retract and check before continuing.")
        raise MotionError(f"The robot finished {name} but is {err_mm:.2f} mm / {err_deg:.2f} deg from the target "
                          f"(tolerance {tol_mm} mm). Was it stopped from the pendant? " + str(result))
    return result


def stop(robot: RobotBackend) -> None:
    robot.send_script(stop_program())
