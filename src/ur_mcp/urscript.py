"""Generate URScript programs for the multiline-script interface."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Literal

_counter = itertools.count(1)


@dataclass
class Segment:
    kind: Literal["movel", "movej_pose", "movej_joints"]
    target: list[float]           # pose [m, rotvec] or joints [rad]
    v: float                      # m/s (movel) or rad/s (movej)
    a: float                      # m/s^2 or rad/s^2
    label: str
    force_limit_n: float | None = None   # guard this segment with the TCP force sensor

    def line(self) -> str:
        if self.kind == "movej_joints":
            return f"movej({fmt_list(self.target)}, a={self.a:.4f}, v={self.v:.4f})"
        fn = "movel" if self.kind == "movel" else "movej"
        return f"{fn}({fmt_pose(self.target)}, a={self.a:.4f}, v={self.v:.4f})"

    def describe(self) -> dict:
        d: dict = {"step": self.label, "command": self.kind.replace("_pose", "").replace("_joints", "")}
        if self.kind == "movej_joints":
            d["joints_deg"] = [round(math.degrees(x), 2) for x in self.target]
            d["speed_deg_s"] = round(math.degrees(self.v), 1)
        else:
            d["tcp_mm"] = [round(x * 1000, 2) for x in self.target[:3]]
            d["rotvec_rad"] = [round(x, 4) for x in self.target[3:]]
            if self.kind == "movel":
                d["speed_mm_s"] = round(self.v * 1000, 1)
            else:
                d["speed_deg_s"] = round(math.degrees(self.v), 1)
        if self.force_limit_n:
            d["force_limit_n"] = self.force_limit_n
        return d


def fmt_list(v: list[float]) -> str:
    return "[" + ", ".join(f"{x:.6f}" for x in v) + "]"


def fmt_pose(p: list[float]) -> str:
    return "p" + fmt_list(p)


def _msg(s: str) -> str:
    return re.sub(r'["\\\n]', "'", s)[:120]


def build_program(segments: list[Segment], title: str) -> tuple[str, str]:
    """Return (program_name, script text). The program runs the segments in order."""
    name = f"mcp_move_{next(_counter)}"
    out = [f"def {name}():", f'  textmsg("MCP: {_msg(title)}")']
    guarded = [s for s in segments if s.force_limit_n]
    if guarded:
        limit = min(s.force_limit_n for s in guarded if s.force_limit_n)
        out += [
            "  thread mcp_force_guard():",
            "    while (True):",
            f"      if (force() > {limit:.2f}):",
            f'        textmsg("MCP: contact force above {limit:.1f} N - motion halted")',
            "        halt",
            "      end",
            "      sync()",
            "    end",
            "  end",
        ]
    for s in segments:
        out.append(f'  textmsg("MCP: {_msg(s.label)}")')
        if s.force_limit_n:
            out += ["  zero_ftsensor()", "  sleep(0.1)", "  mcp_guard = run mcp_force_guard()",
                    "  " + s.line(), "  kill mcp_guard"]
        else:
            out.append("  " + s.line())
    out += [f'  textmsg("MCP: done")', "end"]
    return name, "\n".join(out) + "\n"


def stop_program(decel_m_s2: float = 1.0) -> str:
    """A tiny program that replaces the running one and brings the arm to rest."""
    return f"def mcp_stop():\n  stopl({decel_m_s2:.3f})\nend\n"
