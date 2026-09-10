"""shared_placement.py — the canonical-rotation seam, shared by engine and report.

Three small conversions that BOTH the render path (`shared_engine`) and the write
path (`shared_report`) need, which is why they are neither's private helper: the
engine poses a frame from a candidate, the report serializes the same candidate, and
if those two disagreed the report would describe a pose that was never rendered.

`bpy`-free by contract — `rig.lie` is pure linear algebra, so a placement can be
asserted in the analysis env.
"""

import math

from core import state_json
from rig import lie
from views.sweeps.lib.planner import CANON_DOFS


def canon_quat(c):
    """Candidate (rx, ry, rz) in agent-facing DEGREES -> shared-rotation
    quaternion. The single deg->rad seam for the canonical grammar
    (rig.lie.canonical_rotation stays radians)."""
    return lie.canonical_rotation(*(math.radians(v) for v in c))


def canon_dict(c):
    """(rx,ry,rz) rotation-vector tuple (degrees) -> a rounded {rx,ry,rz} dict.

    Rounded at the same precision as everything else on the wire
    (conventions/state_json.md §5) — the rotation that produced a report's poses is
    part of that report, so it must not print to a different number of places than
    the poses it explains."""
    return {d: state_json.round_component(v) for d, v in zip(CANON_DOFS, c)}


def placement(P0, q_extra, scale=None, centre=None):
    """The right-reoriented pose as a wire-shape pose dict (conventions/state_json.md
    §1), at FULL precision — rounding belongs to the serializer, not here.

    `scale` is passed only on the REPORT path, where an entry must be a
    self-contained placement (state_json.md §4). The render path leaves it None: it
    hands the shared SCALE to `scene.pose_frame` separately, so a scale echoed into
    the pose dict there would be a second copy of the same number.

    `centre` is the canonical-frame pivot the rotation turns about (the frame's
    FK-AABB centre in PER_FRAME mode; None — the canonical origin — in SHARED
    mode). It flows through this seam UNCONDITIONALLY because this function exists
    so the render path and the report path cannot disagree: a pivot applied on one
    side only would be exactly the drift this module was split out to prevent."""
    P = lie.right_reorient(P0, q_extra, centre=centre)
    out = {"quaternion": [float(v) for v in P.q],
           "translation": [float(v) for v in P.t]}
    if scale is not None:
        out["scale"] = float(scale)
    return out
