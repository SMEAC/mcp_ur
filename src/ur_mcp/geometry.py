"""Pose maths in Universal Robots conventions.

A UR pose is ``[x, y, z, rx, ry, rz]``: position in metres plus a rotation
vector (axis * angle, radians). These are the same numbers PolyScope shows,
except that the pendant shows the position in millimetres.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

Pose = list[float]

@dataclass(frozen=True)
class URModel:
    """A UR e-Series arm: nominal DH parameters (Universal Robots, "DH parameters for
    calculations of kinematics and dynamics") and the catalogue reach."""

    name: str              # "ur3e"
    dashboard_name: str    # what Dashboard 'get robot model' answers, e.g. "UR3"
    dh_d: tuple[float, ...]
    dh_a: tuple[float, ...]
    reach_m: float

    @property
    def shoulder_height_m(self) -> float:
        return self.dh_d[0]


_DH_ALPHA = (math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0)

MODELS: dict[str, URModel] = {
    "ur3e": URModel("ur3e", "UR3", (0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921),
                    (0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0), 0.500),
    "ur5e": URModel("ur5e", "UR5", (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
                    (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0), 0.850),
}
MAX_REACH_M = max(m.reach_m for m in MODELS.values())


def get_model(name: str) -> URModel:
    key = name.strip().lower().replace("-", "").replace(" ", "")
    if key in MODELS:
        return MODELS[key]
    for m in MODELS.values():  # dashboard names: "UR3", "UR5"
        if key == m.dashboard_name.lower():
            return m
    raise ValueError(f"Unsupported robot model '{name}'. Supported: {', '.join(MODELS)}")


def _skew(k: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])


def rotvec_to_matrix(rv: Sequence[float]) -> np.ndarray:
    v = np.asarray(rv, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    K = _skew(v / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])  # = 2 sin(th) k
    s2 = float(np.linalg.norm(w))
    th = math.atan2(s2 / 2.0, (np.trace(R) - 1.0) / 2.0)  # accurate near 0 and near 180 deg
    if th < 1e-12:
        return np.zeros(3)
    if s2 > 1e-9:
        return w / s2 * th
    # Exactly (numerically) 180 degrees: axis from the symmetric part, R = 2kk^T - I.
    M = (R + np.eye(3)) / 2.0
    i = int(np.argmax(np.diag(M)))
    k = M[:, i] / math.sqrt(max(M[i, i], 1e-15))
    if np.dot(k, w) < 0:  # keep the sign consistent with whatever antisymmetric part remains
        k = -k
    return k / np.linalg.norm(k) * th


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotvec_to_matrix(pose[3:6])
    T[:3, 3] = pose[:3]
    return T


def matrix_to_pose(T: np.ndarray) -> Pose:
    return [float(x) for x in (*T[:3, 3], *matrix_to_rotvec(T[:3, :3]))]


def pose_trans(a: Sequence[float], b: Sequence[float]) -> Pose:
    """Same as URScript ``pose_trans(a, b)``: b expressed in frame a."""
    return matrix_to_pose(pose_to_matrix(a) @ pose_to_matrix(b))


def pose_inv(p: Sequence[float]) -> Pose:
    return matrix_to_pose(np.linalg.inv(pose_to_matrix(p)))


def position_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a[:3]) - np.asarray(b[:3])))


def orientation_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Angle in radians between the orientations of two poses."""
    R = rotvec_to_matrix(a[3:6]).T @ rotvec_to_matrix(b[3:6])
    return float(np.linalg.norm(matrix_to_rotvec(R)))


def interpolate_pose(a: Sequence[float], b: Sequence[float], s: float) -> Pose:
    """Straight-line position and constant-axis rotation from a (s=0) to b (s=1)."""
    pa, pb = np.asarray(a[:3], float), np.asarray(b[:3], float)
    Ra = rotvec_to_matrix(a[3:6])
    d = matrix_to_rotvec(Ra.T @ rotvec_to_matrix(b[3:6]))
    R = Ra @ rotvec_to_matrix(d * s)
    return [float(x) for x in (*(pa + (pb - pa) * s), *matrix_to_rotvec(R))]


def forward_kinematics(q: Sequence[float], model: URModel | str = "ur3e") -> Pose:
    """Flange pose for joint angles q (radians), standard DH."""
    m = get_model(model) if isinstance(model, str) else model
    T = np.eye(4)
    for th, d, a, al in zip(q, m.dh_d, m.dh_a, _DH_ALPHA):
        ct, st, ca, sa = math.cos(th), math.sin(th), math.cos(al), math.sin(al)
        T = T @ np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ])
    return matrix_to_pose(T)


def mm(pose: Sequence[float]) -> dict[str, float]:
    """Pose as the pendant shows it: mm and radians, rounded."""
    return {
        "x_mm": round(pose[0] * 1000, 3), "y_mm": round(pose[1] * 1000, 3), "z_mm": round(pose[2] * 1000, 3),
        "rx_rad": round(pose[3], 5), "ry_rad": round(pose[4], 5), "rz_rad": round(pose[5], 5),
    }
