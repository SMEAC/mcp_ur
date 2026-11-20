"""Network interfaces of a UR e-Series controller.

* **Script port (30001 primary / 30002 secondary)** - a URScript program sent
  as plain text (``def name(): ... end``) is compiled and run immediately,
  replacing whatever program was running. This is the "Sending Multiline
  Script" method from the Universal Robots forum guide.
* **RTDE (30004)** - robot state streamed at up to 500 Hz. Used to read the
  actual TCP pose and to see when a program starts and stops.
* **Dashboard (29999)** - line-based text commands: power, brakes, modes.

On e-Series robots the script ports only accept programs while the robot is
in **Remote Control** (pendant: profile icon -> Remote Control, after enabling
it under Settings -> System -> Remote Control).
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

log = logging.getLogger(__name__)

ROBOT_MODES = {
    -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING", 3: "POWER_OFF",
    4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE",
}
SAFETY_MODES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
    10: "VALIDATE_JOINT_ID", 11: "UNDEFINED_SAFETY_MODE", 12: "AUTOMATIC_MODE_SAFEGUARD_STOP",
    13: "SYSTEM_THREE_POSITION_ENABLING_STOP",
}
RUNTIME_STATES = {0: "STOPPING", 1: "STOPPED", 2: "PLAYING", 3: "PAUSING", 4: "PAUSED", 5: "RESUMING"}
ROBOT_MODE_RUNNING = 7
SAFETY_NORMAL = (1, 2)  # NORMAL and REDUCED both allow motion


@dataclass
class RobotState:
    timestamp: float = 0.0                        # controller time, s
    received: float = 0.0                         # host monotonic time this sample arrived
    q: list[float] = field(default_factory=lambda: [0.0] * 6)          # rad
    tcp_pose: list[float] = field(default_factory=lambda: [0.0] * 6)   # m, rotvec
    tcp_speed: list[float] = field(default_factory=lambda: [0.0] * 6)  # m/s, rad/s
    target_tcp_pose: list[float] = field(default_factory=lambda: [0.0] * 6)
    tcp_force: list[float] = field(default_factory=lambda: [0.0] * 6)  # N, Nm
    robot_mode: int = -1
    safety_mode: int = 0
    runtime_state: int = 1
    status_bits: int = 0
    speed_scaling: float = 1.0
    speed_fraction: float = 1.0                   # the pendant speed slider

    @property
    def program_running(self) -> bool:
        # robot_status_bits bit 1 = "program running"; runtime_state PLAYING as a fallback
        return bool(self.status_bits & 0b10) or self.runtime_state in (2, 5)

    @property
    def can_move(self) -> bool:
        return self.robot_mode == ROBOT_MODE_RUNNING and self.safety_mode in SAFETY_NORMAL

    @property
    def linear_speed(self) -> float:
        return float(sum(v * v for v in self.tcp_speed[:3]) ** 0.5)

    def describe(self) -> dict:
        return {
            "robot_mode": ROBOT_MODES.get(self.robot_mode, str(self.robot_mode)),
            "safety_mode": SAFETY_MODES.get(self.safety_mode, str(self.safety_mode)),
            "runtime_state": RUNTIME_STATES.get(self.runtime_state, str(self.runtime_state)),
            "program_running": self.program_running,
            "speed_slider_pct": round(self.speed_fraction * 100, 1),
        }


class RobotBackend(Protocol):
    """What the motion layer needs from a robot (real or simulated)."""

    host: str

    def send_script(self, script: str) -> None: ...
    def state(self) -> RobotState: ...
    def dashboard(self, command: str) -> str: ...
    def close(self) -> None: ...


# --------------------------------------------------------------------------
# RTDE
# --------------------------------------------------------------------------

RTDE_REQUEST_PROTOCOL_VERSION = 86   # 'V'
RTDE_TEXT_MESSAGE = 77               # 'M'
RTDE_DATA_PACKAGE = 85               # 'U'
RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS = 79  # 'O'
RTDE_CONTROL_PACKAGE_START = 83      # 'S'
RTDE_CONTROL_PACKAGE_PAUSE = 80      # 'P'

RTDE_TYPES = {
    "VECTOR6D": "6d", "VECTOR3D": "3d", "DOUBLE": "d", "INT32": "i", "UINT32": "I", "UINT64": "Q",
    "UINT8": "B", "BOOL": "?", "VECTOR6INT32": "6i", "VECTOR6UINT32": "6I",
}

RTDE_OUTPUTS = {
    "timestamp": "timestamp",
    "actual_q": "q",
    "actual_TCP_pose": "tcp_pose",
    "actual_TCP_speed": "tcp_speed",
    "target_TCP_pose": "target_tcp_pose",
    "actual_TCP_force": "tcp_force",
    "robot_mode": "robot_mode",
    "safety_mode": "safety_mode",
    "runtime_state": "runtime_state",
    "robot_status_bits": "status_bits",
    "speed_scaling": "speed_scaling",
    "target_speed_fraction": "speed_fraction",
}


class RTDEError(RuntimeError):
    pass


class RTDEReceiver:
    """Minimal RTDE v2 client: subscribes to RTDE_OUTPUTS and keeps the latest sample."""

    def __init__(self, host: str, port: int = 30004, frequency: float = 125.0, timeout_s: float = 5.0):
        self.host, self.port = host, port
        self._lock = threading.Lock()
        self._latest: RobotState | None = None
        self._stop = threading.Event()
        self._error: Exception | None = None
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            self._handshake(frequency)
        except Exception:
            self.sock.close()
            raise
        self._thread = threading.Thread(target=self._run, name="rtde-rx", daemon=True)
        self._thread.start()

    # -- framing --

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise RTDEError("RTDE connection closed by the robot")
            buf += chunk
        return bytes(buf)

    def _send(self, cmd: int, payload: bytes = b"") -> None:
        self.sock.sendall(struct.pack(">HB", len(payload) + 3, cmd) + payload)

    def _recv(self) -> tuple[int, bytes]:
        size, cmd = struct.unpack(">HB", self._recv_exact(3))
        return cmd, self._recv_exact(size - 3)

    def _recv_reply(self, cmd: int) -> bytes:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            c, payload = self._recv()
            if c == cmd:
                return payload
            if c == RTDE_TEXT_MESSAGE:
                log.info("RTDE message: %r", payload)
        raise RTDEError(f"No RTDE reply to command {chr(cmd)!r}")

    def _handshake(self, frequency: float) -> None:
        self._send(RTDE_REQUEST_PROTOCOL_VERSION, struct.pack(">H", 2))
        if not struct.unpack(">B", self._recv_reply(RTDE_REQUEST_PROTOCOL_VERSION))[0]:
            raise RTDEError("Robot refused RTDE protocol version 2 (PolyScope too old?)")
        names = list(RTDE_OUTPUTS)
        self._send(RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS, struct.pack(">d", frequency) + ",".join(names).encode())
        reply = self._recv_reply(RTDE_CONTROL_PACKAGE_SETUP_OUTPUTS)
        self._recipe_id = reply[0]
        types = reply[1:].decode().split(",")
        missing = [n for n, t in zip(names, types) if t not in RTDE_TYPES]
        if missing:
            raise RTDEError(f"RTDE outputs not available on this controller: {missing} ({types})")
        self._names = names
        self._types = types
        self._fmt = ">" + "".join(RTDE_TYPES[t] for t in types)
        self._send(RTDE_CONTROL_PACKAGE_START)
        if not struct.unpack(">B", self._recv_reply(RTDE_CONTROL_PACKAGE_START))[0]:
            raise RTDEError("Robot refused to start RTDE output")

    def _decode(self, payload: bytes) -> RobotState:
        values = struct.unpack(self._fmt, payload[1:])
        st = RobotState(received=time.monotonic())
        i = 0
        for name, t in zip(self._names, self._types):
            n = 6 if t.startswith("VECTOR6") else 3 if t == "VECTOR3D" else 1
            v = values[i:i + n]
            i += n
            setattr(st, RTDE_OUTPUTS[name], list(v) if n > 1 else v[0])
        return st

    def _run(self) -> None:
        self.sock.settimeout(2.0)
        try:
            while not self._stop.is_set():
                cmd, payload = self._recv()
                if cmd == RTDE_DATA_PACKAGE and payload and payload[0] == self._recipe_id:
                    st = self._decode(payload)
                    with self._lock:
                        self._latest = st
                elif cmd == RTDE_TEXT_MESSAGE:
                    log.info("RTDE message: %r", payload)
        except Exception as exc:
            if not self._stop.is_set():
                log.warning("RTDE receiver stopped: %s", exc)
                self._error = exc

    def latest(self, max_age_s: float = 1.0) -> RobotState:
        deadline = time.monotonic() + max_age_s
        while True:
            with self._lock:
                st = self._latest
            if self._error is not None:
                raise RTDEError(f"RTDE link lost: {self._error}. Reconnect with robot_device(operation='connect').")
            if st is not None and time.monotonic() - st.received <= max_age_s:
                return st
            if time.monotonic() > deadline:
                raise RTDEError("No fresh RTDE data from the robot")
            time.sleep(0.01)

    def close(self) -> None:
        self._stop.set()
        try:
            self._send(RTDE_CONTROL_PACKAGE_PAUSE)
        except OSError:
            pass
        self.sock.close()


# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------


class DashboardClient:
    def __init__(self, host: str, port: int = 29999, timeout_s: float = 5.0):
        self.host, self.port, self.timeout_s = host, port, timeout_s
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self.banner = ""

    def _connect(self) -> socket.socket:
        s = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self._file = s.makefile("r", encoding="utf-8", newline="\n")
        self.banner = self._file.readline().strip()  # "Connected: Universal Robots Dashboard Server"
        self._sock = s
        return s

    def command(self, cmd: str) -> str:
        with self._lock:
            for attempt in (1, 2):
                try:
                    s = self._sock or self._connect()
                    s.sendall((cmd.strip() + "\n").encode())
                    line = self._file.readline()
                    if not line:
                        raise ConnectionError("dashboard closed the connection")
                    return line.strip()
                except OSError:
                    self.close()
                    if attempt == 2:
                        raise
            raise AssertionError("unreachable")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None


# --------------------------------------------------------------------------
# Real robot
# --------------------------------------------------------------------------


class URRobot:
    """A real UR e-Series controller reached over Ethernet."""

    def __init__(self, host: str, script_port: int = 30001, rtde_port: int = 30004,
                 dashboard_port: int = 29999, timeout_s: float = 5.0):
        self.host = host
        self.script_port = script_port
        self.timeout_s = timeout_s
        self.dash = DashboardClient(host, dashboard_port, timeout_s)
        try:
            self.dash.command("robotmode")  # fail fast on a wrong IP
        except OSError as exc:
            raise ConnectionError(f"Cannot reach the dashboard server at {host}:{dashboard_port}: {exc}. "
                                  "Check the IP address and that the PC is on the robot's network.") from exc
        self.rtde = RTDEReceiver(host, rtde_port, timeout_s=timeout_s)

    def send_script(self, script: str) -> None:
        """Send a URScript program, exactly as in the forum's multiline example."""
        with socket.create_connection((self.host, self.script_port), timeout=self.timeout_s) as s:
            s.sendall((script.rstrip("\n") + "\n\n").encode())
            time.sleep(0.05)  # let the controller read it before we close

    def state(self) -> RobotState:
        return self.rtde.latest()

    def dashboard(self, command: str) -> str:
        return self.dash.command(command)

    def close(self) -> None:
        try:
            self.rtde.close()
        finally:
            self.dash.close()
