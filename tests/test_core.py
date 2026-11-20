"""Geometry, board file and script generation."""

import json
import math

import numpy as np
import pytest

from ur_mcp import geometry as g
from ur_mcp.board import Board, BoardError, normalise_name
from ur_mcp.motion import plan_home, plan_nudge, plan_retract, plan_to_point
from ur_mcp.session import PROJECT_BOARDS_DIR
from ur_mcp.urscript import build_program

EXAMPLE = PROJECT_BOARDS_DIR / "example_board.json"


@pytest.fixture
def board():
    return Board.load(EXAMPLE)


@pytest.mark.parametrize("rv", [[0, 0, 0], [0.1, -0.2, 0.3], [math.pi, 0, 0], [2.2214, 2.2214, 0],
                                [0, 0, math.pi - 1e-8], [1.0, 2.0, -0.5]])
def test_rotvec_roundtrip(rv):
    R = g.rotvec_to_matrix(rv)
    assert np.allclose(g.rotvec_to_matrix(g.matrix_to_rotvec(R)), R, atol=1e-9)


def test_pose_trans_inverse():
    a = [0.1, -0.2, 0.3, 0.2, 0.4, -1.0]
    b = [0.01, 0.02, 0.03, 3.0, 0.1, 0.0]
    ab = g.pose_trans(a, b)
    back = g.pose_trans(g.pose_inv(a), ab)
    assert g.position_distance(back, b) < 1e-9 and g.orientation_distance(back, b) < 1e-9


def test_fk_zero_pose_ur3e():
    # All joints at zero: arm stretched along -X, flange at a2+a3 and d1-d5 height.
    p = g.forward_kinematics([0] * 6)
    assert p[0] == pytest.approx(-0.24355 - 0.2132, abs=1e-6)
    assert p[1] == pytest.approx(-(0.13105 + 0.0921), abs=1e-6)
    assert p[2] == pytest.approx(0.15185 - 0.08535, abs=1e-6)


def test_names():
    assert normalise_name("Test Point 4") == normalise_name("TP4") == normalise_name("tp_4") == "tp4"


def test_find(board):
    assert board.find("Test Point 4").id == "TP4"
    assert board.find("tp4").id == "TP4"
    assert board.find("4").id == "TP4"
    assert board.find("clock").id == "TP4"
    assert board.find("J3.2").id == "J3.2"
    with pytest.raises(BoardError, match="Available"):
        board.find("TP99")


def test_point_pose_points_down(board):
    tp = board.find("TP4")
    pose = board.point_pose(tp, board.contact_z(tp))
    assert pose[0] * 1000 == pytest.approx(-350 + 55.8, abs=1e-6)
    assert pose[1] * 1000 == pytest.approx(-200 + 42.0, abs=1e-6)
    assert pose[2] * 1000 == pytest.approx(20 - 0.5, abs=1e-6)
    tool_z = g.rotvec_to_matrix(pose[3:])[:, 2]
    assert np.allclose(tool_z, [0, 0, -1], atol=1e-9)
    loc = board.to_local(pose)
    assert loc["x_mm"] == pytest.approx(55.8) and loc["probe_tilt_deg"] == 0


def test_rotated_board_frame():
    d = json.loads(EXAMPLE.read_text())
    d["board_frame"]["rotation_vector_rad"] = [0, 0, math.pi / 2]
    b = Board.from_dict(d)
    tp = b.find("TP1")  # x=12, y=10 -> base: origin + (-10, +12)
    p = b.point_pose(tp, 0)
    assert p[0] * 1000 == pytest.approx(-360, abs=1e-6) and p[1] * 1000 == pytest.approx(-188, abs=1e-6)


def test_validation(board):
    d = json.loads(EXAMPLE.read_text())
    d["motion"]["probe_speed_mm_s"] = 100
    with pytest.raises(BoardError, match="probe_speed"):
        Board.from_dict(d)
    d = json.loads(EXAMPLE.read_text())
    d["test_points"][0]["x_mm"] = 2000
    with pytest.raises(BoardError, match="working range"):
        Board.from_dict(d)
    d = json.loads(EXAMPLE.read_text())
    d["test_points"].append({"id": "TP9", "name": "Test Point 4", "x_mm": 1, "y_mm": 1})
    with pytest.raises(BoardError, match="used by both"):
        Board.from_dict(d)
    d = json.loads(EXAMPLE.read_text())
    d["motion"]["speed"] = 1
    with pytest.raises(BoardError, match="Unknown motion"):
        Board.from_dict(d)


def test_plan_from_home(board):
    home = g.pose_trans(g.forward_kinematics([math.radians(a) for a in board.home_joints_deg]),
                        [0, 0, 0.1, 0, 0, 0])
    plan = plan_to_point(board, home, board.find("TP4"))
    kinds = [s.kind for s in plan.segments]
    assert kinds == ["movej_pose", "movel", "movel"]  # already above clearance: no retract
    assert plan.segments[-1].v == pytest.approx(0.005)
    z = [board.to_local(s.target)["z_mm"] for s in plan.segments]
    assert z == pytest.approx([30.0, 5.0, -0.5])


def test_plan_between_points_retracts_first(board):
    tp3, tp4 = board.find("TP3"), board.find("TP4")
    at_tp3 = board.point_pose(tp3, board.contact_z(tp3))
    plan = plan_to_point(board, at_tp3, tp4)
    assert [s.kind for s in plan.segments] == ["movel"] * 4
    first = board.to_local(plan.segments[0].target)
    assert (first["x_mm"], first["y_mm"], first["z_mm"]) == pytest.approx((40.0, 30.5, 30.0))
    travel = board.to_local(plan.segments[1].target)
    assert (travel["x_mm"], travel["y_mm"], travel["z_mm"]) == pytest.approx((55.8, 42.0, 30.0))


def test_plan_tall_point_and_hover(board):
    j = board.find("J3.2")
    plan = plan_to_point(board, board.point_pose(j, 30), j, stop_above=True)
    assert board.to_local(plan.final_pose)["z_mm"] == pytest.approx(8.5 + 3.0)


def test_retract_noop_above_plane(board):
    assert plan_retract(board, board.local_pose(10, 10, 50)) == []


def test_nudge_limits(board):
    tp = board.find("TP2")
    plan_nudge(board, tp, [1.0, -1.0, 0.0])
    with pytest.raises(BoardError, match="max_nudge"):
        plan_nudge(board, tp, [3.0, 3.0, 0.0])
    with pytest.raises(BoardError, match="deeper"):
        plan_nudge(board, tp, [0, 0, -1.5])


def test_script_text(board):
    d = json.loads(EXAMPLE.read_text())
    d["motion"]["max_force_n"] = 8
    b = Board.from_dict(d)
    plan = plan_to_point(b, b.local_pose(0, 0, 100), b.find("TP2"))
    name, script = build_program(plan.segments, plan.title)
    lines = script.splitlines()
    assert lines[0] == f"def {name}():" and lines[-1] == "end"
    assert "movel(p[" in script and "force() > 8.00" in script
    assert script.count("run mcp_force_guard()") == 1 and "kill mcp_guard" in script
    # balanced def/thread/while/if ... end blocks
    opens = sum(1 for ln in lines if ln.strip().startswith(("def ", "thread ", "while ", "if ")))
    assert opens == sum(1 for ln in lines if ln.strip() == "end")


def test_home_plan_without_board():
    plan = plan_home(None, [0.3, 0, 0.3, 0, 0, 0], [0, -90, 90, -90, -90, 0])
    assert [s.kind for s in plan.segments] == ["movej_joints"]
    assert "movej([0.000000, -1.570796" in plan.segments[0].line()


def test_fk_zero_pose_ur5e():
    p = g.forward_kinematics([0] * 6, "ur5e")
    assert p[0] == pytest.approx(-0.425 - 0.3922, abs=1e-6)
    assert p[2] == pytest.approx(0.1625 - 0.0997, abs=1e-6)


def test_model_lookup():
    assert g.get_model("UR5") is g.MODELS["ur5e"] and g.get_model("UR3e") is g.MODELS["ur3e"]
    with pytest.raises(ValueError, match="Supported"):
        g.get_model("UR10")


def test_reach_depends_on_arm():
    d = json.loads(EXAMPLE.read_text())
    d["limits"].pop("max_reach_mm")
    d["board_frame"]["origin_mm"] = [-600.0, -200.0, 20.0]  # ~640 mm out: UR5e only
    b = Board.from_dict(d)                      # no robot known yet: largest arm's reach
    b.set_robot(g.MODELS["ur5e"])
    assert b.reach_mm == 850
    with pytest.raises(BoardError, match="UR3E"):
        b.set_robot(g.MODELS["ur3e"])
    assert b.robot is g.MODELS["ur5e"]          # unchanged after the refusal
    with pytest.raises(BoardError, match="UR3E"):
        Board.from_dict(d, robot=g.MODELS["ur3e"])
