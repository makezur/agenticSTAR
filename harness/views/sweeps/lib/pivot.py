"""pivot.py — the object's FK-AABB pivot, shared by BOTH sweep grammars.

The camera-frame grammar (sweep/apply, `engine.py`) orbits a frame about a pivot
`c` LEFT-multiplied (x -> dR(x - c) + c in camera space); the canonical grammar
(osweep/oapply, `shared_engine.py`) spins the object about a pivot RIGHT-multiplied
(M' = M @ T(c)·R_extra·T(-c) in the canonical frame). Both pivots are THE SAME
MATERIAL POINT — the canonical-frame AABB centre at the frame's articulation
(core/centre_calculation.py) — which is why the measurement lives here once: two
copies is two definitions of "the object's centre" waiting to disagree.

This module is the Blender-side half (it reads mesh vertices and calls the
renderer's own FK); the arithmetic itself stays in `core.centre_calculation`
(numpy-only) and `rig.transforms.forward_kinematics` (the one FK). What each
grammar DOES with the point stays grammar-local: engine transports it to the
camera (`lie.pose_apply`) for its OrderCtx; shared_engine hands it, still
canonical, to `lie.right_reorient`.
"""

import numpy as np
from mathutils import Matrix

from core import centre_calculation
from rig import transforms
from rig.transforms import _mat_to_np


def canonical_points(o, canonical):
    """One part's points in the CANONICAL frame, as (N,3).

    Mesh VERTICES, not the 8 `bound_box` corners: `bound_box` is the part's LOCAL
    axis-aligned box, so transforming its corners over-bounds the real geometry
    whenever the transform applied to them carries rotation — which an open joint is
    by definition (measured: 0.0066 of centre error at a 45-degree part rotation, on
    an object whose longest dimension is ~1). It is exact only for an axis-aligned
    transform. The pivot is resolved ONCE per run, so reading vertices costs one
    matmul rather than anything per-candidate.

    Falls back to the bbox corners for a part with no vertices (an empty mesh still
    has an 8-corner box), so a degenerate part cannot drop the whole pivot."""
    mesh = getattr(o, "data", None)
    n = len(mesh.vertices) if mesh is not None and hasattr(mesh, "vertices") else 0
    if n:
        flat = np.empty(n * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", flat)
        local = flat.reshape(n, 3)
    else:
        local = np.array([[c[0], c[1], c[2]] for c in o.bound_box], dtype=float)
    return centre_calculation.apply_transform(local, _mat_to_np(canonical[o]))


def canonical_fk(joint_defs, joint_states):
    """Per-part CANONICAL-frame joint transforms at `joint_states`, as numpy 4x4s.

    Reuses `transforms.forward_kinematics` — the one implementation, with the limit
    clamping, the parent-chain accumulation and the cycle check — rather than a
    second copy of the joint arithmetic in `core`: the pivot has to agree with what
    the renderer actually draws, so it must come from the renderer's own FK.

    The IDENTITY base is what makes the result canonical-frame. FK conjugates as
    `base @ A @ base^-1` so that canonical-frame joint declarations act on
    already-posed world matrices; with base = I that collapses to `A`, the pure
    canonical-frame transform, which is the frame `core.centre_calculation` wants
    (articulation is declared in the canonical frame, so the centre is measured
    there and carried to the camera afterwards)."""
    if not joint_defs:
        return {}
    fk = transforms.forward_kinematics(joint_defs, Matrix.Identity(4), joint_states)
    return {part: _mat_to_np(M) for part, M in fk.items()}


def frame_pivot(joint_defs, joint_states, canonical, objs):
    """The canonical-frame FK-AABB pivot for one frame -> (c_canon, wire record).

    THE PIVOT IS A MATERIAL POINT, MEASURED AT THE FRAME'S ARTICULATION: the
    canonical-frame AABB centre at `joint_states` (core.centre_calculation over the
    parts' mesh vertices, articulated by the renderer's own FK). Measuring
    canonically is what makes it rotation-equivariant — a posed-space union AABB
    slides to a different material point as the pose changes (see
    core/centre_calculation.py's module docstring for the measured numbers).

    `c_canon` is CANONICAL-frame: the camera grammar transports it with
    `lie.pose_apply(P0, c_canon)`; the canonical grammar uses it as-is
    (`lie.right_reorient(P0, q, centre=c_canon)`). The record is
    `centre_calculation.pivot_record` sans the camera-frame carry — callers that
    transport the point stamp their own `centre`/`depth_z` (engine does); callers
    that stay canonical record it with the canonical point in both slots."""
    points = {o.name: canonical_points(o, canonical) for o in objs}
    c_canon, _lo, _hi = centre_calculation.canonical_centre(
        points, canonical_fk(joint_defs, joint_states))
    return c_canon, centre_calculation.pivot_record(
        c_canon, c_canon, joint_states)
