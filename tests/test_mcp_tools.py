"""End-to-end through MCP with the simulated robot (sped up)."""

import asyncio
import json
import shutil

import pytest
from fastmcp import Client

from ur_mcp.server import mcp
from ur_mcp.session import PROJECT_BOARDS_DIR, SESSION


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    shutil.copy(PROJECT_BOARDS_DIR / "example_board.json", tmp_path / "example_board.json")
    monkeypatch.setenv("UR_BOARDS_DIR", str(tmp_path))
    monkeypatch.setenv("UR_SIM_SPEEDUP", "40")
    for k in ("UR_HOST", "UR_BOARD_FILE", "UR_ALLOW_RAW_SCRIPT", "UR_HOME_JOINTS_DEG"):
        monkeypatch.delenv(k, raising=False)
    SESSION.disconnect()
    SESSION.board = None
    yield
    SESSION.disconnect()


def run(steps):
    async def go():
        async with Client(mcp) as c:
            await c.call_tool("robot_device", {"operation": "connect", "host": "SIM"})
            await c.call_tool("robot_board", {"operation": "load", "file": "example_board"})
            return await steps(c)
    return asyncio.run(go())


def test_tools_listed():
    async def go():
        async with Client(mcp) as c:
            return {t.name for t in await c.list_tools()}
    assert {"robot_device", "robot_board", "robot_move", "robot_teach", "robot_script"} <= asyncio.run(go())


def test_not_connected():
    async def go():
        async with Client(mcp) as c:
            return await c.call_tool("robot_move", {"operation": "home"}, raise_on_error=False)
    res = asyncio.run(go())
    assert res.is_error and "Not connected" in res.content[0].text


def test_probe_sequence():
    async def steps(c):
        r = await c.call_tool("robot_move", {"operation": "to_point", "point": "Test Point 4"})
        assert r.data["done"] and r.data["point"]["id"] == "TP4"
        assert r.data["position_error_mm"] < 0.1
        assert r.data["tcp_board_mm"]["z_mm"] == pytest.approx(-0.5, abs=0.05)
        assert [s["command"] for s in r.data["plan"]] == ["movej", "movel", "movel"]

        r = await c.call_tool("robot_device", {"operation": "status"})
        assert r.data["resting_on_point"] == "TP4"

        # same point again: nothing to do
        r = await c.call_tool("robot_move", {"operation": "to_point", "point": "tp4"})
        assert r.data["already_there"]

        # next point: retract, travel, descend, probe - all straight lines
        r = await c.call_tool("robot_move", {"operation": "to_point", "point": "TP5"})
        assert [s["step"].split()[0] for s in r.data["plan"]] == ["retract", "travel", "descend", "probe"]
        assert r.data["tcp_board_mm"]["z_mm"] == pytest.approx(-0.8, abs=0.05)

        r = await c.call_tool("robot_move", {"operation": "nudge", "dx_mm": 0.5})
        assert r.data["tcp_board_mm"]["x_mm"] == pytest.approx(73.0, abs=0.05)

        r = await c.call_tool("robot_move", {"operation": "home"})
        assert r.data["joint_error_deg"] < 0.1
        assert r.data["plan"][0]["step"].startswith("retract")
        r = await c.call_tool("robot_teach", {"operation": "where"})
        assert r.data["resting_on_point"] is None and r.data["above_clearance_plane"]
    run(steps)


def test_dry_run_does_not_move():
    async def steps(c):
        before = SESSION.robot.state().tcp_pose
        r = await c.call_tool("robot_move", {"operation": "to_point", "point": "TP1", "dry_run": True})
        assert r.data["dry_run"] and r.data["script"].startswith("def mcp_move_")
        assert SESSION.robot.state().tcp_pose == before
        assert SESSION.robot.scripts == []
    run(steps)


def test_protective_stop_is_reported():
    async def steps(c):
        SESSION.robot.protective_stop()
        r = await c.call_tool("robot_move", {"operation": "to_point", "point": "TP1"}, raise_on_error=False)
        assert r.is_error and "PROTECTIVE_STOP" in r.content[0].text
        await c.call_tool("robot_device", {"operation": "unlock_protective_stop"})
        r = await c.call_tool("robot_move", {"operation": "to_point", "point": "TP1"})
        assert r.data["done"]
    run(steps)


def test_local_mode_rejected():
    async def steps(c):
        SESSION.robot.remote_control = False
        r = await c.call_tool("robot_move", {"operation": "home"}, raise_on_error=False)
        assert r.is_error and "Remote Control" in r.content[0].text
    run(steps)


def test_force_guard_stops_short(tmp_path):
    d = json.loads((tmp_path / "example_board.json").read_text())
    d["motion"]["max_force_n"] = 10
    (tmp_path / "guarded.json").write_text(json.dumps(d))

    async def steps(c):
        await c.call_tool("robot_board", {"operation": "load", "file": "guarded"})
        SESSION.robot.obstacle_z_m = 0.0215  # something 1.5 mm above the pad surface
        r = await c.call_tool("robot_move", {"operation": "to_point", "point": "TP2"}, raise_on_error=False)
        assert r.is_error and "short of the contact" in r.content[0].text
        # the planner no longer assumes it is on a point; the next move retracts first
        SESSION.robot.obstacle_z_m = None
        r = await c.call_tool("robot_move", {"operation": "to_point", "point": "TP3"})
        assert r.data["plan"][0]["step"].startswith("retract")
    run(steps)


def test_teach_frame_and_point(tmp_path):
    async def steps(c):
        r = await c.call_tool("robot_board", {"operation": "create", "file": "new_board", "name": "New"})
        assert r.data["created"].endswith("new_board.json")
        sim = SESSION.robot
        # put the tip at three corners of a board rotated 90 deg about Z, then record a point
        base = sim.state().tcp_pose
        for which, (dx, dy) in {"origin": (0, 0), "x_axis": (0, 0.08), "xy_plane": (-0.05, 0.02)}.items():
            sim._st.tcp_pose = [base[0] + dx, base[1] + dy, 0.03, *base[3:]]
            await c.call_tool("robot_teach", {"operation": "frame_point", "which": which})
        r = await c.call_tool("robot_teach", {"operation": "compute_frame"})
        assert r.data["warnings"] == []
        assert r.data["board_frame"]["rotation_vector_rad"][2] == pytest.approx(1.5708, abs=1e-3)
        sim._st.tcp_pose = [base[0] - 0.01, base[1] + 0.02, 0.03, *base[3:]]
        r = await c.call_tool("robot_teach", {"operation": "record_point", "point": "TP7", "net": "VBAT"})
        assert (r.data["recorded"]["x_mm"], r.data["recorded"]["y_mm"]) == pytest.approx((20.0, 10.0), abs=1e-3)
        saved = json.loads((tmp_path / "new_board.json").read_text())
        assert saved["test_points"][0]["id"] == "TP7"
    run(steps)


def test_raw_script_opt_in(monkeypatch):
    async def steps(c):
        r = await c.call_tool("robot_script", {"script": "textmsg(\"hi\")"}, raise_on_error=False)
        assert r.is_error and "UR_ALLOW_RAW_SCRIPT" in r.content[0].text
        monkeypatch.setenv("UR_ALLOW_RAW_SCRIPT", "1")
        r = await c.call_tool("robot_script", {"script": "def t():\n  movel(p[-0.3,-0.15,0.1,3.14159,0,0], a=1, v=0.5)\nend"})
        assert r.data["finished"]
    run(steps)


def test_resources_and_prompt():
    async def steps(c):
        st = json.loads((await c.read_resource("ur://status"))[0].text)
        b = json.loads((await c.read_resource("ur://board"))[0].text)
        p = await c.get_prompt("probe_test_points", {"points": "TP1, TP4"})
        return st, b, p
    st, b, p = run(steps)
    assert st["connected"] and b["name"].startswith("Example")
    assert "TP1, TP4" in p.messages[0].content.text


def test_board_in_scenario_subfolder(tmp_path, monkeypatch):
    other = tmp_path / "other"
    (other / "TestData" / "Scenario1").mkdir(parents=True)
    src = json.loads((tmp_path / "example_board.json").read_text())
    src["name"] = "Scenario board"
    target = other / "TestData" / "Scenario1" / "scenario_board.json"
    target.write_text(json.dumps(src))
    monkeypatch.setenv("UR_BOARDS_DIR", f"{tmp_path / 'nowhere'};{other / 'TestData'}")

    async def go():
        async with Client(mcp) as c:
            r = await c.call_tool("robot_board", {"operation": "list_files"})
            files = [f["file"] for f in r.data["files"]]
            assert "Scenario1/scenario_board.json" in files and "example_board.json" in files
            r = await c.call_tool("robot_board", {"operation": "load", "file": "scenario_board"})
            assert r.data["file"] == str(target.resolve())
            r = await c.call_tool("robot_board", {"operation": "load", "file": "Scenario1/scenario_board.json"})
            assert r.data["loaded"] == "Scenario board"
            # the bundled example is still found (project boards/ is always searched last)
            r = await c.call_tool("robot_board", {"operation": "load", "file": "example_board"})
            assert r.data["loaded"].startswith("Example")
            # teaching writes back to the scenario folder, not the server project
            await c.call_tool("robot_device", {"operation": "connect", "host": "SIM"})
            await c.call_tool("robot_board", {"operation": "load", "file": "scenario_board"})
            SESSION.robot._st.tcp_pose = SESSION.board.local_pose(15.0, 12.0, 0.0)  # tip on a pad
            await c.call_tool("robot_teach", {"operation": "record_point", "point": "TP99"})
            assert "TP99" in target.read_text()
            (other / "TestData" / "Scenario2").mkdir()
            (other / "TestData" / "Scenario2" / "scenario_board.json").write_text(json.dumps(src))
            r = await c.call_tool("robot_board", {"operation": "load", "file": "scenario_board"},
                                  raise_on_error=False)
            assert r.is_error and "ambiguous" in r.content[0].text
    asyncio.run(go())


def test_no_board_no_motion():
    async def go():
        async with Client(mcp) as c:
            await c.call_tool("robot_device", {"operation": "connect", "host": "SIM"})
            before = SESSION.robot.state().tcp_pose
            out = []
            for args in ({"operation": "to_point", "point": "TP1"}, {"operation": "home"},
                         {"operation": "retract"}):
                out.append(await c.call_tool("robot_move", args, raise_on_error=False))
            return before, out
    before, results = asyncio.run(go())
    for r in results:
        assert r.is_error and "No board loaded" in r.content[0].text
    assert SESSION.robot is None or SESSION.robot.scripts == []


def test_list_files_in_folder(tmp_path):
    ws = tmp_path / "workspace"
    (ws / "TestData" / "Scenario3").mkdir(parents=True)
    (ws / ".mcp.json").write_text("{}")
    (ws / "TestData" / "Scenario3" / "notes.json").write_text('{"not": "a board"}')
    board = json.loads((tmp_path / "example_board.json").read_text())
    target = ws / "TestData" / "Scenario3" / "s3_board.json"
    target.write_text(json.dumps(board))

    async def go():
        async with Client(mcp) as c:
            r = await c.call_tool("robot_board", {"operation": "list_files", "folder": str(ws)})
            assert [f["file"] for f in r.data["files"]] == ["TestData/Scenario3/s3_board.json"]
            r = await c.call_tool("robot_board", {"operation": "load", "file": r.data["files"][0]["path"]})
            assert r.data["file"] == str(target.resolve())
            r = await c.call_tool("robot_board", {"operation": "list_files", "folder": "TestData"},
                                  raise_on_error=False)
            assert r.is_error and "full path" in r.content[0].text
    asyncio.run(go())


def test_ur5e_simulator_and_model_switch(tmp_path):
    d = json.loads((tmp_path / "example_board.json").read_text())
    d["limits"].pop("max_reach_mm")
    d["board_frame"]["origin_mm"] = [-600.0, -200.0, 20.0]
    far = tmp_path / "far_board.json"
    far.write_text(json.dumps(d))

    async def go():
        async with Client(mcp) as c:
            r = await c.call_tool("robot_device", {"operation": "connect", "host": "SIM:UR5E"})
            assert r.data["robot_model"] == "ur5e" and r.data["reach_mm"] == 850
            await c.call_tool("robot_board", {"operation": "load", "file": str(far)})
            r = await c.call_tool("robot_move", {"operation": "to_point", "point": "TP4"})
            assert r.data["done"]
            await c.call_tool("robot_move", {"operation": "home"})
            # the same board can't be used on a UR3e: it is unloaded on connect, not silently kept
            r = await c.call_tool("robot_device", {"operation": "connect", "host": "SIM"})
            assert r.data["robot_model"] == "ur3e" and "board_unloaded" in r.data
            r = await c.call_tool("robot_move", {"operation": "to_point", "point": "TP4"}, raise_on_error=False)
            assert r.is_error and "No board loaded" in r.content[0].text
            r = await c.call_tool("robot_board", {"operation": "load", "file": str(far)}, raise_on_error=False)
            assert r.is_error and "UR3E" in r.content[0].text
    asyncio.run(go())
