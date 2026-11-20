"""The real-robot transport (script socket, RTDE, dashboard) against a fake controller on localhost.

The fake speaks the actual wire protocols and is backed by the simulator, so
URRobot + motion.execute are exercised exactly as they would be on hardware.
"""

import math
import socket
import struct
import threading
import time

import pytest

from ur_mcp import motion
from ur_mcp.board import Board
from ur_mcp.session import PROJECT_BOARDS_DIR
from ur_mcp.simulator import SimulatedUR
from ur_mcp.transport import URRobot

TYPES = {
    "timestamp": "DOUBLE", "actual_q": "VECTOR6D", "actual_TCP_pose": "VECTOR6D",
    "actual_TCP_speed": "VECTOR6D", "target_TCP_pose": "VECTOR6D", "actual_TCP_force": "VECTOR6D",
    "robot_mode": "INT32", "safety_mode": "INT32", "runtime_state": "UINT32",
    "robot_status_bits": "UINT32", "speed_scaling": "DOUBLE", "target_speed_fraction": "DOUBLE",
}


def _listen():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s


class FakeController:
    def __init__(self):
        self.sim = SimulatedUR(speedup=40)
        self.raw_scripts: list[bytes] = []
        self.dash, self.script, self.rtde = _listen(), _listen(), _listen()
        self.stop = threading.Event()
        for fn, srv in ((self._dash, self.dash), (self._script, self.script), (self._rtde, self.rtde)):
            threading.Thread(target=self._accept, args=(srv, fn), daemon=True).start()

    def ports(self):
        return {"dashboard_port": self.dash.getsockname()[1], "script_port": self.script.getsockname()[1],
                "rtde_port": self.rtde.getsockname()[1]}

    def _accept(self, srv, fn):
        while not self.stop.is_set():
            try:
                c, _ = srv.accept()
            except OSError:
                return
            threading.Thread(target=fn, args=(c,), daemon=True).start()

    def _dash(self, c):
        f = c.makefile("r")
        c.sendall(b"Connected: Universal Robots Dashboard Server\n")
        for line in f:
            c.sendall((self.sim.dashboard(line.strip()) + "\n").encode())

    def _script(self, c):
        data = b""
        while chunk := c.recv(65536):
            data += chunk
        self.raw_scripts.append(data)
        self.sim.send_script(data.decode())

    def _rtde(self, c):
        try:
            self._rtde_session(c)
        except (OSError, struct.error):
            pass  # client went away (e.g. it rejected the handshake)

    def _rtde_session(self, c):
        def recv():
            size, cmd = struct.unpack(">HB", c.recv(3, socket.MSG_WAITALL))
            return cmd, c.recv(size - 3, socket.MSG_WAITALL) if size > 3 else b""

        def send(cmd, payload):
            c.sendall(struct.pack(">HB", len(payload) + 3, cmd) + payload)

        cmd, p = recv()
        assert cmd == ord("V") and struct.unpack(">H", p)[0] == 2
        send(ord("V"), b"\x01")
        cmd, p = recv()
        assert cmd == ord("O")
        names = p[8:].decode().split(",")
        send(ord("O"), b"\x07" + ",".join(TYPES.get(n, "NOT_FOUND") for n in names).encode())
        cmd, _ = recv()
        assert cmd == ord("S")
        send(ord("S"), b"\x01")
        send(ord("M"), b"hello")  # stray text message must be ignored
        try:
            while not self.stop.is_set():
                st = self.sim.state()
                data = struct.pack(">Bd6d6d6d6d6diiIIdd", 7, st.timestamp, *st.q, *st.tcp_pose, *st.tcp_speed,
                                   *st.target_tcp_pose, *st.tcp_force, st.robot_mode, st.safety_mode,
                                   st.runtime_state, st.status_bits, st.speed_scaling, st.speed_fraction)
                send(ord("U"), data)
                time.sleep(0.004)
        except OSError:
            pass

    def close(self):
        self.stop.set()
        for s in (self.dash, self.script, self.rtde):
            s.close()


@pytest.fixture
def fake():
    f = FakeController()
    yield f
    f.close()


def test_urrobot_end_to_end(fake):
    robot = URRobot("127.0.0.1", **fake.ports())
    try:
        st = robot.state()
        assert st.robot_mode == 7 and st.tcp_pose == pytest.approx(fake.sim.state().tcp_pose, abs=1e-3)
        assert robot.dashboard("robotmode") == "Robotmode: RUNNING"
        assert robot.dashboard("is in remote control") == "true"

        board = Board.load(PROJECT_BOARDS_DIR / "example_board.json")
        plan = motion.plan_to_point(board, st.tcp_pose, board.find("TP4"))
        res = motion.execute(robot, plan, tol_mm=0.5)
        assert res["position_error_mm"] < 0.1
        # the program went over the script socket as one 'def ... end' block
        text = fake.raw_scripts[-1].decode()
        assert text.startswith("def mcp_move_") and text.rstrip().endswith("end")

        plan = motion.plan_home(board, robot.state().tcp_pose, board.home_joints_deg)
        res = motion.execute(robot, plan, tol_mm=0.5)
        assert res["joint_error_deg"] < 0.1
        assert math.degrees(robot.state().q[1]) == pytest.approx(-90, abs=0.1)
    finally:
        robot.close()


def test_missing_rtde_output_is_reported(fake, monkeypatch):
    monkeypatch.setitem(TYPES, "actual_TCP_force", "NOT_FOUND")
    with pytest.raises(Exception, match="actual_TCP_force"):
        URRobot("127.0.0.1", **fake.ports())


def test_wrong_address():
    s = _listen()
    port = s.getsockname()[1]
    s.close()
    with pytest.raises(ConnectionError, match="dashboard"):
        URRobot("127.0.0.1", dashboard_port=port, timeout_s=1)


def test_model_detected_and_checked(fake, monkeypatch):
    from ur_mcp.session import Session
    s = Session()
    monkeypatch.setattr("ur_mcp.session.URRobot", lambda host, script_port: URRobot(host, **fake.ports()))
    info = s.connect("127.0.0.1")
    assert info["robot_model"] == "ur3e"
    s.disconnect()
    monkeypatch.setenv("UR_MODEL", "ur5e")
    with pytest.raises(RuntimeError, match="UR_MODEL=ur5e but the controller reports a UR3"):
        s.connect("127.0.0.1")
    assert s.robot is None
