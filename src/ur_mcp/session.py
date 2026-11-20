"""Process-wide robot connection, loaded board and probe state."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .board import Board, BoardError
from .geometry import MODELS, URModel, get_model
from .simulator import DEFAULT_SIM_MODEL, SimulatedUR, is_sim, sim_model
from .transport import RobotBackend, URRobot

log = logging.getLogger(__name__)

PROJECT_BOARDS_DIR = Path(__file__).resolve().parents[2] / "boards"


def boards_dirs() -> list[Path]:
    """Directories searched for board files, in order.

    Optional convenience for short names: the client can always pass a full path
    instead. UR_BOARDS_DIR may hold several directories separated by ';' (e.g. a
    test-data folder with one sub-folder per scenario). The project's own boards/
    folder is always searched last, so the bundled example keeps working.
    """
    dirs = [Path(d.strip()).expanduser() for d in os.environ.get("UR_BOARDS_DIR", "").split(";") if d.strip()]
    if PROJECT_BOARDS_DIR not in dirs:
        dirs.append(PROJECT_BOARDS_DIR)
    return dirs


def boards_dir() -> Path:
    """Where new board files are created: the first search directory."""
    return boards_dirs()[0]


_SKIP_DIRS = {"node_modules", "__pycache__", "venv", "site-packages"}
MAX_SEARCH_DEPTH = 4


def is_board_file(f: Path) -> bool:
    """A JSON file with a top-level 'board_frame' (cheap check, small files only)."""
    try:
        if f.stat().st_size > 1_000_000:
            return False
        d = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return isinstance(d, dict) and "board_frame" in d


def _walk_json(root: Path, depth: int = 0):
    try:
        entries = sorted(root.iterdir())
    except OSError:
        return
    for e in entries:
        if e.name.startswith("."):
            continue
        if e.is_dir():
            if depth < MAX_SEARCH_DEPTH and e.name.lower() not in _SKIP_DIRS:
                yield from _walk_json(e, depth + 1)
        elif e.suffix.lower() == ".json":
            yield e


def find_board_files(dirs: list[Path] | None = None) -> list[tuple[Path, Path]]:
    """(search dir, file) for every board JSON under the given (default: search) directories,
    sub-folders included up to MAX_SEARCH_DEPTH levels."""
    out: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for d in dirs if dirs is not None else boards_dirs():
        if not d.is_dir():
            continue
        for f in _walk_json(d):
            r = f.resolve()
            if r not in seen and is_board_file(f):
                seen.add(r)
                out.append((d, f))
    return out


class Session:
    def __init__(self) -> None:
        self.lock = threading.RLock()            # connection / board state
        self.motion_lock = threading.Lock()      # one move at a time
        self.abort = threading.Event()           # set by stop to end a wait early
        self.robot: RobotBackend | None = None
        self.model: URModel | None = None         # arm type of the connected robot
        self.board: Board | None = None
        self.current_point: str | None = None    # test point the probe is resting on
        self.hover_point: str | None = None      # test point the probe is hovering over
        self.nudge_mm: list[float] = [0.0, 0.0, 0.0]
        self.frame_points: dict[str, list[float]] = {}
        self.last_move: dict[str, Any] | None = None

    @property
    def simulated(self) -> bool:
        return isinstance(self.robot, SimulatedUR)

    def clear_position(self) -> None:
        self.current_point, self.hover_point, self.nudge_mm = None, None, [0.0, 0.0, 0.0]

    # ---- connection -----------------------------------------------------

    def connect(self, host: str | None = None, script_port: int | None = None) -> dict[str, Any]:
        with self.lock:
            self.disconnect()
            host = host or os.environ.get("UR_HOST")
            if not host:
                raise RuntimeError("No robot address. Give host='192.168.x.y' (or 'SIM'), or set UR_HOST.")
            configured = os.environ.get("UR_MODEL") or None
            if is_sim(host):
                model = get_model(sim_model(host) or configured or DEFAULT_SIM_MODEL)
                robot: RobotBackend = SimulatedUR(speedup=float(os.environ.get("UR_SIM_SPEEDUP", "1")),
                                                  model=model)
            else:
                port = script_port or int(os.environ.get("UR_SCRIPT_PORT", "30001"))
                robot = URRobot(host, script_port=port)
                try:
                    model = self._detect_model(robot, configured)
                except Exception:
                    robot.close()
                    raise
            self.robot, self.model = robot, model
            self.clear_position()
            info: dict[str, Any] = {"connected": True, "host": robot.host, "simulated": self.simulated,
                                    "robot_model": model.name, "reach_mm": model.reach_m * 1000}
            for key, cmd in (("serial", "get serial number"), ("polyscope", "PolyscopeVersion"),
                             ("remote_control", "is in remote control")):
                try:
                    info[key] = robot.dashboard(cmd)
                except Exception as exc:  # older PolyScope lacks some commands
                    info[key] = f"n/a ({exc})"
            info.update(robot.state().describe())
            if self.board is not None:
                try:
                    self.board.set_robot(model)
                except BoardError as exc:
                    info["board_unloaded"] = (f"{self.board.name} was unloaded: it doesn't fit the "
                                              f"{model.name.upper()}. {exc}")
                    self.board = None
            return info

    @staticmethod
    def _detect_model(robot: RobotBackend, configured: str | None) -> URModel:
        """Ask the controller which arm it drives; UR_MODEL (bench config) must agree if set."""
        try:
            reported = robot.dashboard("get robot model").strip()
            detected = get_model(reported)
        except Exception as exc:
            if configured:
                log.warning("Could not read the robot model (%s); using UR_MODEL=%s", exc, configured)
                return get_model(configured)
            raise RuntimeError(f"Could not determine the robot model ({exc}). Set UR_MODEL to one of "
                               f"{', '.join(MODELS)} in the server environment.") from exc
        if configured and get_model(configured) != detected:
            raise RuntimeError(f"UR_MODEL={configured} but the controller reports a {reported}. "
                               "Fix UR_MODEL or check the robot's IP address.")
        return detected

    def disconnect(self) -> bool:
        with self.lock:
            if self.robot is None:
                return False
            try:
                self.robot.close()
            finally:
                self.robot, self.model = None, None
                self.clear_position()
            return True

    def require(self) -> RobotBackend:
        with self.lock:
            if self.robot is None:
                if os.environ.get("UR_HOST"):
                    self.connect()
                else:
                    raise RuntimeError("Not connected. Call robot_device(operation='connect', host=...) "
                                       "first (host='SIM' for the simulator).")
            assert self.robot is not None
            return self.robot

    # ---- board ----------------------------------------------------------

    def resolve_board_path(self, path: str) -> Path:
        p = Path(path).expanduser()
        with_ext = [p] if p.suffix.lower() == ".json" else [p, Path(f"{p}.json")]
        if p.is_absolute():
            candidates = with_ext
        else:
            candidates = [d / c for d in boards_dirs() for c in with_ext] + [Path.cwd() / c for c in with_ext]
        for c in candidates:
            if c.is_file():
                return c.resolve()
        if not p.is_absolute():
            # A bare name may live in a sub-folder, e.g. Scenario1/scenario1_opamp_board.json
            names = {c.name.lower() for c in with_ext}
            hits = [f for _, f in find_board_files() if f.name.lower() in names]
            if len(hits) == 1:
                return hits[0].resolve()
            if len(hits) > 1:
                raise BoardError(f"Board name '{path}' is ambiguous: {[str(h) for h in hits]}. "
                                 "Give the sub-folder or the full path.")
        raise BoardError(f"Board file '{path}' not found. Searched (with sub-folders): "
                         f"{[str(d) for d in boards_dirs()]}. Set UR_BOARDS_DIR or give a full path.")

    def load_board(self, path: str) -> Board:
        with self.lock:
            board = Board.load(self.resolve_board_path(path), self.model)
            self.board = board
            self.clear_position()
            return board

    def require_board(self) -> Board:
        """The board loaded in this session. Deliberately no default: the robot must never move
        using the coordinates of a board nobody chose."""
        with self.lock:
            if self.board is None:
                raise RuntimeError(
                    "No board loaded. Load the board that is in the fixture first: "
                    "robot_board(operation='load', file=<full path to the board JSON>). "
                    "robot_board(operation='list_files', folder=...) lists the board files in a folder.")
            return self.board


SESSION = Session()
