"""PCB description: board frame, test points and motion settings, loaded from JSON.

Coordinates of test points are in **millimetres in the board frame**. The board
frame is a UR pose (as shown on the teach pendant: X/Y/Z in mm, RX/RY/RZ as a
rotation vector in radians) whose XY plane is the top surface of the PCB and
whose +Z axis points up, away from the board, towards the robot.

At a test point the probe (the active TCP) points straight down into the
board: the tool Z axis is the board's -Z axis. ``tool_rz_deg`` turns the probe
about its own axis, which is useful for routing the probe cable.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import (
    MAX_REACH_M,
    Pose,
    URModel,
    matrix_to_pose,
    pose_to_matrix,
    rot_x,
    rot_z,
)

# Hard ceilings. A board file may lower these but not exceed them.
SPEED_CAPS: dict[str, float] = {
    "travel_speed_mm_s": 250.0,
    "travel_accel_mm_s2": 1500.0,
    "joint_speed_deg_s": 90.0,
    "joint_accel_deg_s2": 180.0,
    "descend_speed_mm_s": 100.0,
    "probe_speed_mm_s": 20.0,
    "probe_accel_mm_s2": 500.0,
    "retract_speed_mm_s": 100.0,
}


class BoardError(ValueError):
    pass


@dataclass
class MotionSettings:
    clearance_height_mm: float = 30.0      # travel plane, board frame Z; must clear every component
    approach_height_mm: float = 5.0        # "above the point" = point Z + this; slow probing starts here
    contact_depth_mm: float = 0.5          # how far below the pad surface the tip is driven (spring probe)
    travel_speed_mm_s: float = 100.0
    travel_accel_mm_s2: float = 500.0
    joint_speed_deg_s: float = 45.0        # movej (home, and travel from far away)
    joint_accel_deg_s2: float = 90.0
    descend_speed_mm_s: float = 30.0       # clearance plane -> approach height
    probe_speed_mm_s: float = 5.0          # approach height -> contact ("moves down slowly")
    probe_accel_mm_s2: float = 100.0
    retract_speed_mm_s: float = 30.0       # straight up off the board
    travel_motion: str = "auto"            # auto | movel | movej
    max_force_n: float | None = None       # stop the slow descent above this TCP force (e-Series F/T sensor)
    position_tolerance_mm: float = 1.0     # arrival check against RTDE actual TCP pose
    max_nudge_mm: float = 3.0              # largest manual correction away from a nominal test point

    def validate(self) -> None:
        for name, cap in SPEED_CAPS.items():
            v = getattr(self, name)
            if not (0 < v <= cap):
                raise BoardError(f"motion.{name}={v} must be > 0 and <= {cap}")
        if self.travel_motion not in ("auto", "movel", "movej"):
            raise BoardError("motion.travel_motion must be 'auto', 'movel' or 'movej'")
        if self.clearance_height_mm <= 0 or self.approach_height_mm <= 0:
            raise BoardError("clearance_height_mm and approach_height_mm must be positive")
        if self.contact_depth_mm < 0 or self.contact_depth_mm > 5:
            raise BoardError("contact_depth_mm must be between 0 and 5 mm")
        if self.max_force_n is not None and not (1 <= self.max_force_n <= 50):
            raise BoardError("max_force_n must be between 1 and 50 N (or null to disable)")
        if not (0 < self.position_tolerance_mm <= 10):
            raise BoardError("position_tolerance_mm must be in (0, 10]")
        if not (0 <= self.max_nudge_mm <= 10):
            raise BoardError("max_nudge_mm must be between 0 and 10")


@dataclass
class TestPoint:
    __test__ = False  # not a pytest class

    id: str
    x_mm: float
    y_mm: float
    z_mm: float = 0.0
    name: str = ""
    net: str = ""
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    contact_depth_mm: float | None = None
    approach_height_mm: float | None = None
    tool_rz_deg: float | None = None

    def info(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [])}


def normalise_name(s: str) -> str:
    """'Test Point 4', 'TP4', 'tp 4', 'testpoint_4' -> 'tp4'."""
    n = re.sub(r"[^a-z0-9]", "", str(s).lower())
    return re.sub(r"^testpoint", "tp", n)


def _floats(v: Any, n: int, what: str) -> list[float]:
    try:
        out = [float(x) for x in v]
    except Exception as exc:
        raise BoardError(f"{what} must be a list of {n} numbers") from exc
    if len(out) != n or not all(math.isfinite(x) for x in out):
        raise BoardError(f"{what} must be a list of {n} finite numbers")
    return out


@dataclass
class Board:
    name: str
    frame: Pose                                  # board frame in robot base, metres / rotvec
    points: list[TestPoint]
    motion: MotionSettings
    tool_rz_deg: float = 0.0
    home_joints_deg: list[float] | None = None
    max_reach_mm: float | None = None            # optional board limit, tighter than the arm's own reach
    min_base_z_mm: float | None = None
    robot: URModel | None = None                 # the connected arm; None until a robot is connected
    path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # ---- loading / saving ----------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: Path | None = None, robot: URModel | None = None) -> Board:
        d = copy.deepcopy(d)
        bf = d.get("board_frame")
        if not isinstance(bf, dict):
            raise BoardError("'board_frame' with 'origin_mm' and 'rotation_vector_rad' is required")
        origin = _floats(bf.get("origin_mm"), 3, "board_frame.origin_mm")
        rv = _floats(bf.get("rotation_vector_rad", [0, 0, 0]), 3, "board_frame.rotation_vector_rad")
        frame = [origin[0] / 1000, origin[1] / 1000, origin[2] / 1000, *rv]

        known = {f.name for f in fields(MotionSettings)}
        mraw = d.get("motion", {}) or {}
        unknown = set(mraw) - known
        if unknown:
            raise BoardError(f"Unknown motion settings: {sorted(unknown)}. Known: {sorted(known)}")
        motion = MotionSettings(**mraw)
        motion.validate()

        probe = d.get("probe", {}) or {}
        if "contact_depth_mm" in probe:  # convenience location
            motion.contact_depth_mm = float(probe["contact_depth_mm"])
            motion.validate()

        tp_known = {f.name for f in fields(TestPoint)}
        points: list[TestPoint] = []
        seen: dict[str, str] = {}
        for i, p in enumerate(d.get("test_points", [])):
            if not isinstance(p, dict) or "id" not in p:
                raise BoardError(f"test_points[{i}] needs at least 'id', 'x_mm', 'y_mm'")
            extra = set(p) - tp_known
            if extra:
                raise BoardError(f"test_points[{i}] ({p['id']}): unknown keys {sorted(extra)}")
            try:
                tp = TestPoint(**{**p, "x_mm": float(p["x_mm"]), "y_mm": float(p["y_mm"]),
                                  "z_mm": float(p.get("z_mm", 0.0))})
            except KeyError as exc:
                raise BoardError(f"test_points[{i}] ({p['id']}): missing {exc}") from exc
            for key in {tp.id, tp.name, *tp.aliases} - {""}:
                k = normalise_name(key)
                if k in seen and seen[k] != tp.id:
                    raise BoardError(f"Name '{key}' is used by both {seen[k]} and {tp.id}")
                seen[k] = tp.id
            points.append(tp)

        home = d.get("home", {}) or {}
        home_j = _floats(home["joints_deg"], 6, "home.joints_deg") if "joints_deg" in home else None
        limits = d.get("limits", {}) or {}
        board = cls(
            name=str(d.get("name", path.stem if path else "board")),
            frame=frame,
            points=points,
            motion=motion,
            tool_rz_deg=float(probe.get("tool_rz_deg", 0.0)),
            home_joints_deg=home_j,
            max_reach_mm=None if limits.get("max_reach_mm") is None else float(limits["max_reach_mm"]),
            min_base_z_mm=None if limits.get("min_base_z_mm") is None else float(limits["min_base_z_mm"]),
            robot=robot,
            path=path,
            raw=d,
        )
        board.check_points()
        return board

    @classmethod
    def load(cls, path: str | Path, robot: URModel | None = None) -> Board:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise BoardError(f"Board file not found: {p}")
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BoardError(f"{p.name}: invalid JSON: {exc}") from exc
        return cls.from_dict(d, p, robot)

    def save(self, path: str | Path | None = None) -> Path:
        p = Path(path).expanduser().resolve() if path else self.path
        if p is None:
            raise BoardError("No file name: give a path to save to")
        p.write_text(json.dumps(self.raw, indent=2) + "\n", encoding="utf-8")
        self.path = p
        return p

    def check_points(self) -> None:
        """Every target this board can produce must be inside the robot's limits."""
        m = self.motion
        for tp in self.points:
            top = tp.z_mm + self.approach_height(tp)
            if tp.z_mm + 1.0 > m.clearance_height_mm:
                raise BoardError(f"{tp.id}: z_mm={tp.z_mm} is not below clearance_height_mm="
                                 f"{m.clearance_height_mm} - raise the clearance height")
            for z in (self.contact_z(tp), min(top, m.clearance_height_mm)):
                self.check_reachable(self.local_pose(tp.x_mm, tp.y_mm, z, self.tool_rz(tp)), tp.id)

    # ---- lookup ----------------------------------------------------------

    def find(self, name: str) -> TestPoint:
        key = normalise_name(name)
        if key.isdigit():
            key = "tp" + key
        for tp in self.points:
            if key in {normalise_name(n) for n in (tp.id, tp.name, *tp.aliases) if n}:
                return tp
        raise BoardError(f"Unknown test point '{name}'. Available: "
                         + ", ".join(f"{tp.id}" + (f" ({tp.name})" if tp.name else "") for tp in self.points))

    def contact_depth(self, tp: TestPoint) -> float:
        return self.motion.contact_depth_mm if tp.contact_depth_mm is None else tp.contact_depth_mm

    def approach_height(self, tp: TestPoint) -> float:
        return self.motion.approach_height_mm if tp.approach_height_mm is None else tp.approach_height_mm

    def tool_rz(self, tp: TestPoint | None) -> float:
        return self.tool_rz_deg if tp is None or tp.tool_rz_deg is None else tp.tool_rz_deg

    def contact_z(self, tp: TestPoint) -> float:
        return tp.z_mm - self.contact_depth(tp)

    # ---- frames ----------------------------------------------------------

    def local_pose(self, x_mm: float, y_mm: float, z_mm: float, tool_rz_deg: float = 0.0) -> Pose:
        """Base-frame TCP pose for a probe tip at (x, y, z) mm in the board frame, pointing down."""
        L = np.eye(4)
        L[:3, :3] = rot_x(math.pi) @ rot_z(math.radians(tool_rz_deg))
        L[:3, 3] = (x_mm / 1000, y_mm / 1000, z_mm / 1000)
        return matrix_to_pose(pose_to_matrix(self.frame) @ L)

    def point_pose(self, tp: TestPoint, z_mm: float) -> Pose:
        return self.local_pose(tp.x_mm, tp.y_mm, z_mm, self.tool_rz(tp))

    def to_local(self, pose: Pose) -> dict[str, float]:
        """Express a base-frame TCP pose in board coordinates (mm)."""
        T = np.linalg.inv(pose_to_matrix(self.frame)) @ pose_to_matrix(pose)
        tool_z = T[:3, 2]
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -tool_z[2]))))
        tool_x = T[:3, 0]
        return {
            "x_mm": round(float(T[0, 3]) * 1000, 3),
            "y_mm": round(float(T[1, 3]) * 1000, 3),
            "z_mm": round(float(T[2, 3]) * 1000, 3),
            "probe_tilt_deg": round(tilt, 2),  # 0 = pointing straight down into the board
            "tool_rz_deg": round(math.degrees(math.atan2(-tool_x[1], tool_x[0])), 2),
        }

    def normal(self) -> np.ndarray:
        """Board +Z (up, away from the PCB) in base coordinates."""
        return pose_to_matrix(self.frame)[:3, 2]

    @property
    def reach_mm(self) -> float:
        """Usable reach: the connected arm's catalogue reach (before connecting: the largest
        supported arm's), further limited by limits.max_reach_mm when the board sets it."""
        arm = (self.robot.reach_m if self.robot else MAX_REACH_M) * 1000
        return min(arm, self.max_reach_mm) if self.max_reach_mm else arm

    def set_robot(self, robot: URModel | None) -> None:
        """Switch to another arm, re-checking every point against its reach first."""
        previous = self.robot
        self.robot = robot
        try:
            self.check_points()
        except BoardError:
            self.robot = previous
            raise

    def check_reachable(self, pose: Pose, what: str = "target") -> None:
        reach = self.reach_mm
        shoulder_mm = (self.robot.shoulder_height_m if self.robot else 0.15) * 1000
        r = math.hypot(pose[0], pose[1]) * 1000
        if math.sqrt(r * r + (pose[2] * 1000 - shoulder_mm) ** 2) > reach or r < 60:
            arm = self.robot.name.upper() if self.robot else "largest supported arm (robot not connected yet)"
            raise BoardError(f"{what}: position {[round(v * 1000, 1) for v in pose[:3]]} mm is outside the "
                             f"working range of the {arm}: reach {reach:.0f} mm, and not over the base. "
                             "Check board_frame and the point coordinates.")
        if self.min_base_z_mm is not None and pose[2] * 1000 < self.min_base_z_mm:
            raise BoardError(f"{what}: Z {pose[2] * 1000:.1f} mm is below limits.min_base_z_mm={self.min_base_z_mm}")

    # ---- editing ---------------------------------------------------------

    def set_frame(self, frame: Pose) -> None:
        self.raw.setdefault("board_frame", {})
        self.raw["board_frame"]["origin_mm"] = [round(v * 1000, 3) for v in frame[:3]]
        self.raw["board_frame"]["rotation_vector_rad"] = [round(v, 6) for v in frame[3:6]]
        updated = Board.from_dict(self.raw, self.path, self.robot)
        self.__dict__.update(updated.__dict__)

    def upsert_point(self, point_id: str, x_mm: float, y_mm: float, z_mm: float, **extra: Any) -> TestPoint:
        entry = {"id": point_id, "x_mm": round(x_mm, 3), "y_mm": round(y_mm, 3), "z_mm": round(z_mm, 3)}
        entry.update({k: v for k, v in extra.items() if v not in (None, "")})
        raw = copy.deepcopy(self.raw)
        pts = raw.setdefault("test_points", [])
        for i, p in enumerate(pts):
            if normalise_name(p.get("id", "")) == normalise_name(point_id):
                pts[i] = {**p, **entry, "id": p["id"]}
                break
        else:
            pts.append(entry)
        updated = Board.from_dict(raw, self.path, self.robot)  # validates before committing
        self.__dict__.update(updated.__dict__)
        return self.find(point_id)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file": str(self.path) if self.path else None,
            "board_frame": {"origin_mm": [round(v * 1000, 3) for v in self.frame[:3]],
                            "rotation_vector_rad": [round(v, 6) for v in self.frame[3:]]},
            "robot_model": self.robot.name if self.robot else None,
            "reach_mm": self.reach_mm,
            "home_joints_deg": self.home_joints_deg,
            "motion": asdict(self.motion),
            "test_points": [tp.info() for tp in self.points],
        }
