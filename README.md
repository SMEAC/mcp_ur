# ur-mcp

An MCP server that lets AI assistants move an **oscilloscope probe held by a Universal Robots
e-Series arm (UR3e or UR5e)** onto named test points of a PCB. It is the companion to
[msox2014a-mcp](https://github.com/SMEAC/mcp_msox2014a): one server positions the probe, the
other measures.

A person places the PCB in a fixed position (a fixture). A JSON **board file** lists the test
points in board coordinates. The assistant can then say "probe Test Point 4". The robot lifts the
probe, moves above TP4, descends, and lowers the probe slowly onto the pad.

The robot is programmed **online**. Every move is sent as a complete URScript program
(`def … end`) over the controller's script port. This is the *Sending Multiline Script* method
from the Universal Robots forum guide
[UR remote control via TCP/IP](https://forum.universal-robots.com/t/guide-and-example-ur-remote-control-via-tcpip/32973).
RTDE reports back when the program has finished and where the probe actually is.

A built-in **simulator** (`SIM`, or `SIM:UR5E`) runs the same scripts, so you can try everything
without the robot.

## Supported robots

| Arm | Reach | Detected as |
|---|---|---|
| UR3e | 500 mm | Dashboard `get robot model` → `UR3` |
| UR5e | 850 mm | Dashboard `get robot model` → `UR5` |

The server asks the controller which arm it is on `connect`. Every target is checked against that
arm's reach; a board file can set a tighter `limits.max_reach_mm`. A board that doesn't fit the
connected arm is refused when it loads, and if you switch to a smaller arm with a board loaded,
that board is unloaded. Set `UR_MODEL` (bench configuration) to be sure you're on the arm you
think you are: the server refuses to connect if the controller reports another model.

Everything else (URScript, RTDE, dashboard, motion logic) is the same on both. Other e-Series arms
(UR10e, UR16e, …) only need an entry in `MODELS` in [geometry.py](src/ur_mcp/geometry.py).

## How a probe move works

```text
          travel (movel at clearance height)
   ┌────────────────────────────────────────┐
   │ retract                                │  descend (30 mm/s)
   │ (straight up)                          ▼
   │                                 approach height (5 mm above pad)
   │                                        │  probe (5 mm/s, optional force guard)
 ──●── TP3 ─────────── PCB top surface ──── ● TP4 ── (pad Z − contact depth)
```

1. **Retract**: if the tip is below the clearance plane, it goes straight up along the board
   normal. The orientation doesn't change.
2. **Travel**: the probe moves to above the next point at clearance height, pointing straight
   down. Between points this is a straight `movel` in the clearance plane. Coming from further
   away, such as from home, it is a `movej`.
3. **Descend**: down to `approach_height_mm` above the pad.
4. **Probe**: slowly down to `pad Z − contact_depth_mm`, which compresses a spring-loaded tip a
   little. The arm's built-in force/torque sensor can optionally guard this step.

All four steps go to the robot as **one program**. The server then watches RTDE until the program
has finished, and checks that the TCP is within `position_tolerance_mm` of the target. The tool
call returns only after the robot has stopped, so the next tool call (for example a scope capture)
happens with the probe in place.

Home works the same way: retract first, then `movej` to the home joint angles.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). There are no robot-specific
dependencies; the RTDE and dashboard clients are built in.

```powershell
git clone https://github.com/SMEAC/mcp_ur.git
cd mcp_ur
uv sync
uv run ur-mcp --simulate   # smoke run (stdio; Ctrl+C to exit)
uv run pytest
```

## Preparing the robot

| Step | Where | Why |
|---|---|---|
| Give the robot a fixed IP on the same subnet as the PC | Pendant: Settings → System → Network | The server connects to ports 29999, 30001 and 30004 |
| **Enable Remote Control** | Pendant: Settings → System → Remote Control → Enable; then profile icon (top right) → **Remote Control** | e-Series robots ignore scripts sent over the network in Local mode. The server checks this and tells you. |
| **Set the TCP to the probe tip** and make it the default TCP | Installation → General → TCP | All positions are positions of the active TCP. If the TCP is the flange, the probe tip lands in the wrong place. |
| Set the payload (probe + holder mass) | Installation → General → Payload | Needed for correct motion and for the force guard |
| Configure safety limits | Installation → Safety | Speed and force limits, plus safety planes (e.g. a plane just under the board), are your real protection |
| Allow these ports through the Windows firewall (outbound) | PC | |

The server never changes the installation. It only sends motion programs and dashboard commands.

## Board file

Board files are JSON. See [boards/example_board.json](boards/example_board.json). Keep each real
board's file with its PCB and test data, wherever that is, for example
`TestData\Scenario1\scenario1_opamp_board.json` next to the PCB and schematic files.
`robot_teach` writes changes back to the file that was loaded.

### Choosing the board: at runtime, never by default

Which board is in the fixture is part of the task, not of the server's setup, so it is chosen at
runtime with a tool call. There is no default board. The server refuses every move, home
included, until a board has been loaded in the session, so it never moves the robot using the
coordinates of a board nobody chose.

```text
robot_board(operation="list_files", folder="C:/dev/mcp_stuff/mcp_test/TestData")   # discover
robot_board(operation="load", file="C:/dev/mcp_stuff/mcp_test/TestData/Scenario1/scenario1_opamp_board.json")
```

Use **full paths**. The server runs in its own folder (`uv run --directory …`), so a relative path
would not mean what the client means by it. The client (Claude Code) knows its workspace and
writes the full path itself. Its `CLAUDE.md` can say where the scenarios are.

`list_files` only lists JSON files that contain a `board_frame`. It searches sub-folders up to four
levels deep, skipping hidden folders and `node_modules`.

As an optional shortcut, `UR_BOARDS_DIR` names folders where short names are looked up, so that
`file="scenario1_opamp_board"` works. The project's `boards/` folder (with `example_board`) is
always searched last. If two sub-folders hold a file with the same name, the server asks for the
sub-folder.

> MCP *roots* (the client telling the server its workspace folders) would be the protocol-level
> way to do this, but roots are deprecated from the 2026-07-28 MCP spec (SEP-2577) and FastMCP no
> longer exposes them, so this server doesn't use them.

```json
{
  "name": "Example PCB rev A",
  "board_frame": {
    "origin_mm": [-350.0, -200.0, 20.0],
    "rotation_vector_rad": [0.0, 0.0, 0.0]
  },
  "probe": { "tool_rz_deg": 0.0 },
  "motion": {
    "clearance_height_mm": 30.0,
    "approach_height_mm": 5.0,
    "contact_depth_mm": 0.5,
    "probe_speed_mm_s": 5.0,
    "max_force_n": null
  },
  "home": { "joints_deg": [0.0, -90.0, 90.0, -90.0, -90.0, 0.0] },
  "limits": { "max_reach_mm": 480.0, "min_base_z_mm": 10.0 },
  "test_points": [
    { "id": "TP4", "name": "Test Point 4", "x_mm": 55.8, "y_mm": 42.0, "z_mm": 0.0,
      "net": "CLK_8MHZ", "description": "MCU clock", "aliases": ["clock"] },
    { "id": "J3.2", "name": "Header J3 pin 2", "x_mm": 88.0, "y_mm": 20.0, "z_mm": 8.5,
      "approach_height_mm": 3.0, "contact_depth_mm": 0.2 }
  ]
}
```

**Board frame.** The board frame is a pose in robot base coordinates, in the same numbers the
pendant shows: X/Y/Z in mm and RX/RY/RZ as a rotation vector in radians. Its XY plane is the
**top surface of the PCB** and its +Z axis points up, away from the board. A pad on the surface
therefore has `z_mm = 0`; raised features such as header pins have a positive `z_mm`. The probe
always points along −Z of the board, and `tool_rz_deg` turns it about its own axis to route the
cable.

**Test point names.** Names are matched loosely: `"TP4"`, `"tp 4"`, `"Test Point 4"`, `"4"` and
any `aliases` all find the same point.

**Per-point overrides.** A point can set its own `contact_depth_mm`, `approach_height_mm` and
`tool_rz_deg`.

**Motion settings** (all optional; the defaults are shown in the example):

| Setting | Default | Meaning |
|---|---|---|
| `clearance_height_mm` | 30 | Travel plane above the board surface. **Must clear the tallest component** (and the probe body!) |
| `approach_height_mm` | 5 | Height above the pad where slow probing starts |
| `contact_depth_mm` | 0.5 | How far below the pad surface the tip is driven (spring compression), 0–5 |
| `travel_speed_mm_s` / `travel_accel_mm_s2` | 100 / 500 | `movel` travel in the clearance plane (cap 250 mm/s) |
| `joint_speed_deg_s` / `joint_accel_deg_s2` | 45 / 90 | `movej` moves: home, and travel from far away (cap 90 °/s) |
| `descend_speed_mm_s` | 30 | Clearance plane → approach height |
| `probe_speed_mm_s` / `probe_accel_mm_s2` | 5 / 100 | Approach height → contact, "moves down slowly" (cap 20 mm/s) |
| `retract_speed_mm_s` | 30 | Lifting off the board |
| `travel_motion` | `auto` | `auto`, `movel` or `movej` for the travel step |
| `max_force_n` | `null` (off) | Halt the slow descent if the TCP force exceeds this (1–50 N) |
| `position_tolerance_mm` | 1.0 | Arrival check against the RTDE actual TCP pose |
| `max_nudge_mm` | 3 | Largest manual correction allowed away from a nominal point |

When a board file loads, every point is checked: contact, approach and travel poses must be
within the connected arm's reach (before connecting: the largest supported arm's), above
`limits.min_base_z_mm`, and below the clearance plane. A bad file is rejected before anything
moves.

### Teaching a board instead of measuring it

If the fixture position isn't known from CAD, teach it with the probe:

1. `robot_board(operation="create", file="C:/…/TestData/Scenario2/my_board.json")`
2. Jog the probe tip (pendant, freedrive) to the board origin corner on the top surface, then call
   `robot_teach(operation="frame_point", which="origin")`.
3. Jog to a point along the board's +X edge (≥ 20 mm away), then call `which="x_axis"`.
4. Jog to any point on the +Y side, then call `which="xy_plane"`.
5. `robot_teach(operation="compute_frame")`: the frame is written to the file.
6. For each test point, jog the tip onto the pad and call
   `robot_teach(operation="record_point", point="TP4", name="Test Point 4", net="+3V3")`.

Test points from the PCB CAD (pick-and-place or test-point report) can be typed straight into
`test_points` instead. They stay valid whenever the frame is re-taught, because they are in board
coordinates. Remember to add a `home` block.

## Register with an MCP client

**Claude Code**

```powershell
claude mcp add ur -e UR_HOST=192.168.1.20 -- uv run --directory C:\dev\mcp_stuff\mcp_ur ur-mcp
# simulator:
claude mcp add ur-sim -- uv run --directory C:\dev\mcp_stuff\mcp_ur ur-mcp --simulate
```

**`.mcp.json`**, together with the oscilloscope (see [.mcp.json.example](.mcp.json.example)):

```json
{
  "mcpServers": {
    "ur": {
      "command": "uv",
      "args": ["run", "--directory", "C:\\dev\\mcp_stuff\\mcp_ur", "ur-mcp"],
      "env": { "UR_HOST": "192.168.1.20", "UR_MODEL": "ur3e" }
    },
    "msox2014a": {
      "command": "uv",
      "args": ["run", "--directory", "C:\\dev\\mcp_stuff\\mcp_msox2014a", "msox2014a-mcp"],
      "env": { "MSOX_RESOURCE": "USB0::0x0957::0x1798::MY56271763::0::INSTR" }
    }
  }
}
```

## Configuration

Environment variables (and CLI flags) are **bench configuration**: they describe the rig and are
fixed for as long as the server process runs. Anything that changes from test to test, such as
which board is in the fixture, is a tool parameter instead.

| Env var | Default | Purpose |
|---|---|---|
| `UR_HOST` | *(unset)* | Robot IP or hostname to auto-connect on first use. `SIM` selects the simulated UR3e, `SIM:UR5E` a simulated UR5e. |
| `UR_MODEL` | *(unset)* | Optional: `ur3e` or `ur5e`. On a real robot it must match what the controller reports (or it's used if the controller can't say). For `SIM` it picks the simulated arm. |
| `UR_BOARDS_DIR` | *(unset)* | Optional: folders where short board names are looked up, separated by `;` (sub-folders included). The project's `boards/` is always searched last. Not needed when the client passes full paths. |
| `UR_SCRIPT_PORT` | `30001` | URScript port: 30001 (primary, as in the forum example) or 30002 (secondary) |
| `UR_HOME_JOINTS_DEG` | *(unset)* | Home position if the board file has none, e.g. `0,-90,90,-90,-90,0` |
| `UR_ALLOW_RAW_SCRIPT` | *(unset)* | `1` enables the `robot_script` tool |
| `UR_SIM_SPEEDUP` | `1` | Simulator time factor |
| `UR_LOG_LEVEL` | `WARNING` | Log level. Logs go to stderr. |

CLI flags: `--robot <ip>|SIM`, `--simulate`, `--transport {stdio,http,sse}`,
`--host`, `--port` (HTTP bind address and port, default 10941), `--log-level`.

## Tools

| Tool | Operations |
|---|---|
| `robot_device` | `connect`, `disconnect`, `status`, `power_on`, `brake_release`, `power_off`, `unlock_protective_stop`, `close_popup`, `dashboard` (raw Dashboard Server command) |
| `robot_board` | `list_files` (optionally in a `folder`), `load` (required before any move), `info`, `points`, `point`, `create` |
| `robot_move` | `to_point`, `above_point` (stop at approach height to check alignment), `retract`, `home`, `nudge` (small correction while on a point, max `max_nudge_mm`), `stop`; any move with `dry_run=true` returns the plan and the URScript without moving |
| `robot_teach` | `where`, `record_point`, `frame_point`, `compute_frame` |
| `robot_script` | Raw URScript. **Disabled** unless `UR_ALLOW_RAW_SCRIPT=1`; it bypasses every check |

Resources: `ur://status` and `ur://board`. Prompt: `probe_test_points`, which probes a list
of points and measures each one with the oscilloscope server.

### Example session

```text
robot_device(operation="connect", host="192.168.1.20")
robot_device(operation="status")                    # RUNNING, NORMAL, remote_control true?
robot_board(operation="load", file="C:/…/TestData/Scenario1/scenario1_opamp_board.json")
robot_move(operation="to_point", point="TP4", dry_run=true)   # look before you leap
robot_move(operation="above_point", point="TP4")    # hover 5 mm above: check by eye
robot_move(operation="to_point", point="TP4")       # down onto the pad
scope_capture(operation="single", channels="1")     # (oscilloscope server)
scope_measure(operation="all")
robot_move(operation="to_point", point="Test Point 5")   # lifts, travels, probes
robot_move(operation="home")
```

## Safety

The server lowers the risk of an error. It does **not** make the cell safe. The robot's safety
configuration and the e-stop do that.

- Every move retracts to the clearance plane before going sideways, even after a restart, a stop,
  or a pendant jog. The decision uses the **actual** TCP position from RTDE, not remembered
  state.
- Speeds are capped (probe 20 mm/s, travel 250 mm/s, joints 90 °/s). Every target is checked
  against reach and `min_base_z_mm` before anything is sent.
- Before every move the server checks robot mode, safety mode, Remote Control, and that no other
  program is running. During a move it aborts on any protective or safeguard stop.
- After every move it checks the position against RTDE. If the force guard stopped the probe
  short of the pad, the tool returns an error and says so.
- `robot_move(operation="stop")` replaces the running program with `stopl`. It is a convenience,
  not a safety function.
- `nudge` is limited to `max_nudge_mm` from the nominal point, and to 1 mm deeper than the
  contact depth.
- On a new setup, use `dry_run`, then `above_point`, and keep a hand on the e-stop before the
  first `to_point`. Try a reduced speed slider on the pendant too; the server's timeouts scale
  with it.

## Project layout

```text
src/ur_mcp/
  server.py     FastMCP tools, resources, prompt
  motion.py     move planning (retract / travel / descend / probe), execution and arrival checks
  board.py      board JSON: frame, test points, motion settings, validation, teaching updates
  urscript.py   URScript program generation (def ... end, force-guard thread)
  transport.py  script socket (30001), RTDE client (30004), Dashboard client (29999)
  simulator.py  simulated UR3e / UR5e that runs the generated scripts
  geometry.py   UR pose maths (rotation vectors, pose_trans), arm models (DH, reach), kinematics
  session.py    connection, loaded board, probe state
boards/         board files
tests/          unit tests, MCP end-to-end tests, and a fake UR controller on localhost that
                speaks the real RTDE/dashboard/script protocols
```

## Status and limitations

- **Not yet run on a real UR3e or UR5e.** The real transport is tested against a fake controller that
  speaks the RTDE v2, dashboard and script-socket protocols, and all motion logic runs against
  the simulator. Test on hardware with `dry_run`, a reduced speed slider and the e-stop ready.
  Check in particular:
  - that RTDE reports `program_running` for scripts sent to port 30001 (the server relies on
    this to detect the end of a move);
  - the force guard (`max_force_n`), which uses a URScript thread with `force()` and `halt`. Try
    it with a low speed and a soft target first. It is off by default.
- The probe always approaches perpendicular to the board. Tilted approaches are not supported.
- The path from home to the first point uses `movej`, whose TCP path is not a straight line. Home
  should be above the clearance plane, with nothing tall in between.
- The simulator has no collision model, and it moves `movej`-to-pose in a straight line.
- One board is loaded at a time. Fixture repeatability is assumed; there is no vision or fiducial
  alignment.

## License

MIT - see [LICENSE](LICENSE).
