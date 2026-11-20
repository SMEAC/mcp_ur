"""FastMCP server: move an oscilloscope probe on a UR e-Series arm (UR3e / UR5e) to named PCB test points."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import anyio
import numpy as np
from fastmcp import FastMCP
from pydantic import Field

from . import __version__, motion
from .board import Board, BoardError
from .geometry import matrix_to_rotvec, mm
from .motion import MotionError, Plan
from .session import SESSION, boards_dir, boards_dirs, find_board_files
from .urscript import build_program

INSTRUCTIONS = """\
Moves an oscilloscope probe held by a Universal Robots e-Series arm (UR3e or
UR5e) onto named test points of a PCB. The PCB sits in a fixed position; its
test points are listed in a JSON board file in board coordinates (mm). The arm
model is read from the robot on connect; reach checks use it.

Typical flow: robot_device(operation='connect', host=<robot IP>, or 'SIM' /
'SIM:UR5E' for the simulator) ->
robot_board(operation='load', file=<full path of the board JSON>) ->
robot_move(operation='to_point', point='TP4') -> measure with the oscilloscope ->
robot_move(operation='to_point', point='TP5') ... -> robot_move(operation='home').
No board is loaded by default: every session must load the board that is
actually in the fixture before the robot will move (home included).

Every move first lifts the probe straight up to the clearance plane, travels
above the next point, descends and then probes slowly. Use dry_run=True to see
the planned motion without moving. robot_move(operation='stop') halts motion
but is NOT a safety function; the pendant e-stop is.
"""

mcp = FastMCP("ur", instructions=INSTRUCTIONS, version=__version__)


def _j(v: Any) -> Any:
    """Make numpy scalars JSON-friendly."""
    if isinstance(v, dict):
        return {k: _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _position() -> dict[str, Any]:
    robot = SESSION.require()
    st = robot.state()
    out: dict[str, Any] = {
        "tcp_base": mm(st.tcp_pose),
        "joints_deg": [round(math.degrees(q), 3) for q in st.q],
        "resting_on_point": SESSION.current_point,
        "hovering_over_point": SESSION.hover_point,
    }
    if SESSION.current_point and any(SESSION.nudge_mm):
        out["nudge_mm"] = SESSION.nudge_mm
    if SESSION.board is not None:
        b = SESSION.board
        loc = b.to_local(st.tcp_pose)
        out["tcp_board_mm"] = loc
        out["above_clearance_plane"] = loc["z_mm"] >= b.motion.clearance_height_mm - 0.5
        if b.points:
            near = min(b.points, key=lambda p: math.hypot(p.x_mm - loc["x_mm"], p.y_mm - loc["y_mm"]))
            out["nearest_point"] = {"id": near.id,
                                    "xy_distance_mm": round(math.hypot(near.x_mm - loc["x_mm"], near.y_mm - loc["y_mm"]), 2),
                                    "height_above_pad_mm": round(loc["z_mm"] - near.z_mm, 2)}
    return out


# --------------------------------------------------------------------------
# robot_device
# --------------------------------------------------------------------------

_DASHBOARD_OPS = {
    "power_on": "power on",
    "brake_release": "brake release",
    "power_off": "power off",
    "unlock_protective_stop": "unlock protective stop",
    "close_popup": "close safety popup",
}


@mcp.tool(annotations={"title": "Robot connection and power"})
def robot_device(
    operation: Annotated[
        Literal["connect", "disconnect", "status", "power_on", "brake_release", "power_off",
                "unlock_protective_stop", "close_popup", "dashboard"],
        Field(description="connect | disconnect | status: modes, TCP position, probe state | power_on | "
                          "brake_release (the arm moves slightly when the brakes release) | power_off | "
                          "unlock_protective_stop (only after finding the cause) | close_popup: close a "
                          "safety popup | dashboard: send any Dashboard Server command"),
    ],
    host: Annotated[str | None, Field(description="connect: robot IP address or hostname, or 'SIM' for the "
                                                  "simulator ('SIM:UR5E' for a simulated UR5e). Omit to use "
                                                  "$UR_HOST.")] = None,
    script_port: Annotated[int | None, Field(description="connect: URScript port, 30001 (primary, default) "
                                                         "or 30002 (secondary)")] = None,
    command: Annotated[str | None, Field(description="dashboard: the command, e.g. 'robotmode', "
                                                     "'safetystatus', 'PolyscopeVersion'")] = None,
) -> dict[str, Any]:
    """Connect to the robot (UR3e / UR5e) and manage power, brakes and safety state."""
    if operation == "connect":
        return SESSION.connect(host, script_port)
    if operation == "disconnect":
        return {"disconnected": SESSION.disconnect()}
    if operation == "status" and SESSION.robot is None and not os.environ.get("UR_HOST"):
        return {"connected": False, "hint": "robot_device(operation='connect', host='<robot IP>' or 'SIM')",
                "board": SESSION.board.name if SESSION.board else None}
    robot = SESSION.require()
    if operation == "status":
        st = robot.state()
        return _j({"connected": True, "host": robot.host, "simulated": SESSION.simulated,
                   "robot_model": SESSION.model.name if SESSION.model else None,
                   **st.describe(), "remote_control": robot.dashboard("is in remote control"),
                   "board": SESSION.board.name if SESSION.board else None,
                   "move_in_progress": SESSION.motion_lock.locked(), **_position()})
    if operation == "dashboard":
        if not command:
            raise ValueError("dashboard needs command=...")
        return {"command": command, "reply": robot.dashboard(command)}
    reply = robot.dashboard(_DASHBOARD_OPS[operation])
    if operation in ("power_off", "unlock_protective_stop"):
        SESSION.clear_position()
    return {"operation": operation, "reply": reply}


# --------------------------------------------------------------------------
# robot_board
# --------------------------------------------------------------------------


@mcp.tool(annotations={"title": "PCB board file and test points", "readOnlyHint": False})
def robot_board(
    operation: Annotated[
        Literal["list_files", "load", "info", "points", "point", "create"],
        Field(description="list_files: board files in `folder` (or, without it, in UR_BOARDS_DIR and the "
                          "project's boards/), sub-folders included | load: load the board that is in the "
                          "fixture - required before any move | "
                          "info: frame, motion settings and points | points: test point list | "
                          "point: one point, with its robot coordinates | create: new, empty board file to "
                          "fill with robot_teach"),
    ],
    file: Annotated[str | None, Field(description="load/create: full path of the board JSON (preferred), e.g. "
                                                  "'C:/bench/TestData/Scenario1/scenario1_opamp_board.json'. "
                                                  "Short names such as 'scenario1_opamp_board' are looked up in "
                                                  "UR_BOARDS_DIR (sub-folders too) and the project's boards/; "
                                                  ".json may be left out")] = None,
    folder: Annotated[str | None, Field(description="list_files: folder to search, e.g. the workspace's "
                                                    "TestData folder (full path)")] = None,
    point: Annotated[str | None, Field(description="point: test point id or name, e.g. 'TP4' or "
                                                   "'Test Point 4'")] = None,
    name: Annotated[str | None, Field(description="create: board name")] = None,
) -> dict[str, Any]:
    """Load the PCB description (board frame + test points) and look up test points."""
    if operation == "list_files":
        if folder:
            base = Path(folder).expanduser()
            if not base.is_absolute() or not base.is_dir():
                raise BoardError(f"folder must be an existing folder given as a full path, got '{folder}'")
            dirs = [base]
        else:
            dirs = boards_dirs()
        files = [{"file": f.relative_to(d).as_posix(), "path": str(f)} for d, f in find_board_files(dirs)]
        return {"searched": [str(d) for d in dirs], "files": files,
                "loaded": str(SESSION.board.path) if SESSION.board else None}
    if operation == "load":
        if not file:
            raise ValueError("load needs file=...")
        b = SESSION.load_board(file)
        return {"loaded": b.name, "file": str(b.path), "test_points": [p.id for p in b.points]}
    if operation == "create":
        if not file:
            raise ValueError("create needs file=...")
        path = boards_dir() / file if not os.path.isabs(file) else file
        path = str(path) if str(path).lower().endswith(".json") else f"{path}.json"
        if os.path.exists(path):
            raise BoardError(f"{path} already exists - load it instead")
        b = Board.from_dict({"name": name or os.path.splitext(os.path.basename(path))[0],
                             "board_frame": {"origin_mm": [0, 0, 0], "rotation_vector_rad": [0, 0, 0]},
                             "probe": {"tool_rz_deg": 0}, "motion": {}, "test_points": []})
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        b.save(path)
        SESSION.board = b
        SESSION.clear_position()
        return {"created": str(b.path), "next": "Teach the board frame: robot_teach(operation='frame_point', "
                                               "which='origin' / 'x_axis' / 'xy_plane') then 'compute_frame'."}
    b = SESSION.require_board()
    if operation == "info":
        return _j(b.summary())
    if operation == "points":
        return {"board": b.name, "test_points": [p.info() for p in b.points]}
    if operation == "point":
        if not point:
            raise ValueError("point needs point=...")
        tp = b.find(point)
        return _j({**tp.info(), "contact_depth_mm": b.contact_depth(tp),
                   "approach_height_mm": b.approach_height(tp),
                   "contact_pose_base": mm(b.point_pose(tp, b.contact_z(tp)))})
    raise ValueError(f"Unknown operation {operation}")


# --------------------------------------------------------------------------
# robot_move
# --------------------------------------------------------------------------


def _home_joints() -> list[float]:
    b = SESSION.board
    if b is not None and b.home_joints_deg:
        return b.home_joints_deg
    env = os.environ.get("UR_HOME_JOINTS_DEG")
    if env:
        vals = [float(x) for x in env.replace(";", ",").split(",")]
        if len(vals) == 6:
            return vals
    raise BoardError("No home position: add \"home\": {\"joints_deg\": [...6 values...]} to the board file "
                     "or set UR_HOME_JOINTS_DEG.")


def _move(operation: str, point: str | None, offset: list[float], dry_run: bool) -> dict[str, Any]:
    robot = SESSION.require()
    if not SESSION.motion_lock.acquire(blocking=False):
        raise MotionError("Another move is in progress. Wait for it, or call robot_move(operation='stop').")
    try:
        SESSION.abort.clear()
        st = robot.state()
        board = SESSION.require_board()  # even home: it must retract clear of the board first
        tp = None
        if operation in ("to_point", "above_point"):
            if not point:
                raise ValueError(f"{operation} needs point=... (e.g. 'TP4')")
            tp = board.find(point)
            if (operation == "to_point" and SESSION.current_point == tp.id and not any(SESSION.nudge_mm)
                    and motion.at_pose(st, board.point_pose(tp, board.contact_z(tp)),
                                       board.motion.position_tolerance_mm)[0]):
                return {"point": tp.id, "already_there": True, **_position()}
            plan = motion.plan_to_point(board, st.tcp_pose, tp, stop_above=operation == "above_point")
        elif operation == "retract":
            segs = motion.plan_retract(board, st.tcp_pose)
            if not segs:
                SESSION.current_point = None
                return {"retracted": False, "reason": "already at or above the clearance plane", **_position()}
            plan = Plan("retract", segs, segs[-1].target)
        elif operation == "home":
            plan = motion.plan_home(board, st.tcp_pose, _home_joints())
        elif operation == "nudge":
            if SESSION.current_point is None:
                raise MotionError("nudge only works while the probe is on a test point (after to_point)")
            tp = board.find(SESSION.current_point)
            total = [a + b for a, b in zip(SESSION.nudge_mm, offset)]
            plan = motion.plan_nudge(board, tp, total)
        else:
            raise ValueError(f"Unknown operation {operation}")

        if dry_run:
            name, script = build_program(plan.segments, plan.title)
            return _j({"dry_run": True, "plan": plan.describe(), "script": script})

        tol = board.motion.position_tolerance_mm
        try:
            res = motion.execute(robot, plan, tol, SESSION.abort)
        except Exception:
            SESSION.clear_position()
            raise
        SESSION.clear_position()
        if operation == "to_point":
            SESSION.current_point = tp.id
        elif operation == "above_point":
            SESSION.hover_point = tp.id
        elif operation == "nudge":
            SESSION.current_point, SESSION.nudge_mm = tp.id, total
        out: dict[str, Any] = {"operation": operation, "done": True, "plan": plan.describe(), **res}
        if tp is not None:
            out["point"] = tp.info()
        out["tcp_board_mm"] = board.to_local(robot.state().tcp_pose)
        SESSION.last_move = out
        return _j(out)
    finally:
        SESSION.motion_lock.release()


@mcp.tool(annotations={"title": "Move the probe", "destructiveHint": True})
async def robot_move(
    operation: Annotated[
        Literal["to_point", "above_point", "retract", "home", "nudge", "stop"],
        Field(description="to_point: lift, travel above the point, descend and probe it slowly | "
                          "above_point: as to_point but stop at approach height above the pad (check "
                          "alignment) | retract: straight up to the clearance plane | home: retract, then "
                          "move to the home joint position | nudge: small correction while on a point | "
                          "stop: halt the current motion now"),
    ],
    point: Annotated[str | None, Field(description="Test point id or name for to_point / above_point, "
                                                   "e.g. 'TP4', 'Test Point 4' or '4'")] = None,
    dx_mm: Annotated[float, Field(description="nudge: board X correction, mm")] = 0.0,
    dy_mm: Annotated[float, Field(description="nudge: board Y correction, mm")] = 0.0,
    dz_mm: Annotated[float, Field(description="nudge: board Z correction, mm (negative = press deeper)")] = 0.0,
    dry_run: Annotated[bool, Field(description="Return the planned steps and URScript without moving")] = False,
) -> dict[str, Any]:
    """Move the oscilloscope probe between PCB test points and home. Blocks until the robot has stopped."""
    if operation == "stop":
        robot = SESSION.require()
        SESSION.abort.set()
        motion.stop(robot)
        SESSION.clear_position()
        return {"stopped": True, "note": "Probe position is now unknown to the planner; the next move "
                                         "starts with a retract if the tip is below the clearance plane."}
    return await anyio.to_thread.run_sync(_move, operation, point, [dx_mm, dy_mm, dz_mm], dry_run)


# --------------------------------------------------------------------------
# robot_teach
# --------------------------------------------------------------------------


def _unit(v: np.ndarray, what: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 0.02:
        raise BoardError(f"{what}: the taught points are too close together (need >= 20 mm)")
    return v / n


@mcp.tool(annotations={"title": "Teach board frame and test points"})
def robot_teach(
    operation: Annotated[
        Literal["where", "record_point", "frame_point", "compute_frame"],
        Field(description="where: current TCP in base and board coordinates | record_point: store the "
                          "current probe-tip position as a test point (jog the tip onto the pad with the "
                          "pendant first) | frame_point: remember the tip position as 'origin', 'x_axis' or "
                          "'xy_plane' | compute_frame: set the board frame from those three points"),
    ],
    point: Annotated[str | None, Field(description="record_point: test point id, e.g. 'TP4'")] = None,
    name: Annotated[str | None, Field(description="record_point: readable name, e.g. 'Test Point 4'")] = None,
    net: Annotated[str | None, Field(description="record_point: net name, e.g. '+3V3'")] = None,
    description: Annotated[str | None, Field(description="record_point: free text")] = None,
    which: Annotated[Literal["origin", "x_axis", "xy_plane"] | None, Field(
        description="frame_point: origin = board origin corner on the PCB top surface; x_axis = a point "
                    "along the board +X edge; xy_plane = any point on the +Y side of the X edge")] = None,
    save: Annotated[bool, Field(description="record_point / compute_frame: write the board file")] = True,
) -> dict[str, Any]:
    """Teach by demonstration: read where the probe is and turn it into board coordinates."""
    robot = SESSION.require()
    if operation == "where":
        return _j(_position())
    st = robot.state()
    if operation == "frame_point":
        if which is None:
            raise ValueError("frame_point needs which='origin' | 'x_axis' | 'xy_plane'")
        SESSION.frame_points[which] = list(st.tcp_pose)
        return {"stored": which, "tcp_base": mm(st.tcp_pose),
                "have": sorted(SESSION.frame_points), "need": ["origin", "x_axis", "xy_plane"]}
    board = SESSION.require_board()
    if operation == "record_point":
        if not point:
            raise ValueError("record_point needs point=... (e.g. 'TP4')")
        loc = board.to_local(st.tcp_pose)
        warnings = []
        if loc["probe_tilt_deg"] > 5:
            warnings.append(f"The probe is tilted {loc['probe_tilt_deg']} deg from vertical; the stored "
                            "point uses the tip position, and probing will be done vertically.")
        tp = board.upsert_point(point, loc["x_mm"], loc["y_mm"], loc["z_mm"],
                                name=name, net=net, description=description)
        path = str(board.save()) if save else None
        return {"recorded": tp.info(), "saved_to": path, "warnings": warnings}
    if operation == "compute_frame":
        fp = SESSION.frame_points
        missing = {"origin", "x_axis", "xy_plane"} - set(fp)
        if missing:
            raise BoardError(f"Teach these first with frame_point: {sorted(missing)}")
        o, px, py = (np.asarray(fp[k][:3]) for k in ("origin", "x_axis", "xy_plane"))
        x = _unit(px - o, "x_axis")
        z = _unit(np.cross(x, py - o), "xy_plane")
        if abs(float(np.dot(_unit(py - o, "xy_plane"), x))) > 0.98:
            raise BoardError("xy_plane point is (almost) on the X axis line - pick one well to the side")
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])
        frame = [*o.tolist(), *matrix_to_rotvec(R).tolist()]
        warnings = []
        if z[2] < 0.5:
            warnings.append("Board +Z points downwards or steeply sideways. With a PCB lying on the table, "
                            "the xy_plane point must be on the left of the X direction (seen from above).")
        board.set_frame(frame)
        path = str(board.save()) if save else None
        SESSION.clear_position()
        return _j({"board_frame": board.summary()["board_frame"], "saved_to": path, "warnings": warnings})
    raise ValueError(f"Unknown operation {operation}")


# --------------------------------------------------------------------------
# robot_script (opt-in)
# --------------------------------------------------------------------------


@mcp.tool(annotations={"title": "Send raw URScript (opt-in)", "destructiveHint": True})
async def robot_script(
    script: Annotated[str, Field(description="URScript: one line, or a full 'def name(): ... end' program")],
    wait: Annotated[bool, Field(description="Wait for the program to finish")] = True,
    timeout_s: Annotated[float, Field(description="wait: give up after this many seconds")] = 60.0,
) -> dict[str, Any]:
    """Send URScript directly. No clearance or reach checks! Disabled unless UR_ALLOW_RAW_SCRIPT=1."""
    if os.environ.get("UR_ALLOW_RAW_SCRIPT") != "1":
        raise PermissionError("Raw URScript is disabled. Set UR_ALLOW_RAW_SCRIPT=1 in the server environment "
                              "to enable it. It bypasses every safety check of the probe planner.")
    robot = SESSION.require()

    def go() -> dict[str, Any]:
        with SESSION.motion_lock:
            st = robot.state()
            motion.check_ready(robot, st)
            SESSION.clear_position()
            robot.send_script(script)
            if not wait:
                return {"sent": True}
            import time
            t0 = time.monotonic()
            seen = False
            while time.monotonic() - t0 < timeout_s:
                s = robot.state()
                if s.program_running:
                    seen = True
                elif seen or time.monotonic() - t0 > 3.0:
                    return {"sent": True, "finished": seen, "duration_s": round(time.monotonic() - t0, 2),
                            **s.describe(), "tcp_base": mm(s.tcp_pose)}
                time.sleep(0.01)
            return {"sent": True, "finished": False, "note": "still running after timeout"}

    return _j(await anyio.to_thread.run_sync(go))


# --------------------------------------------------------------------------
# resources and prompts
# --------------------------------------------------------------------------


@mcp.resource("ur://status", mime_type="application/json")
def status_resource() -> str:
    if SESSION.robot is None:
        return json.dumps({"connected": False})
    return json.dumps(_j({"connected": True, **SESSION.robot.state().describe(), **_position()}))


@mcp.resource("ur://board", mime_type="application/json")
def board_resource() -> str:
    return json.dumps(_j(SESSION.board.summary()) if SESSION.board else {"loaded": False})


@mcp.prompt()
def probe_test_points(points: str, measurement: str = "Vpp, mean and frequency") -> str:
    """Probe a list of test points one after another and measure each with the oscilloscope."""
    return (
        f"Probe these PCB test points in order: {points}.\n"
        "1. robot_device(operation='status') - check the robot is RUNNING, safety NORMAL, Remote Control on, "
        "and a board is loaded (robot_board(operation='info')).\n"
        "2. For each point: robot_move(operation='to_point', point=<id>); when it reports done, capture with "
        f"the oscilloscope MCP (scope_capture single) and record {measurement} (scope_measure).\n"
        "3. If a move fails, stop: report the error and do not continue to the next point.\n"
        "4. Finish with robot_move(operation='home') and give a table: point, net, measurements."
    )
