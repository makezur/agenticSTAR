"""centre_calculation.py — the object CENTRE used as the camera-frame sweep pivot.

The camera-frame order grammar (roll/yaw/pitch, see rig/lie.apply_order) orbits the
object about a PIVOT `c`: x -> dR(x - c) + c. This module decides which point that is,
in the CANONICAL frame at a given articulation.

Measuring canonically and transporting the point afterwards is what makes the pivot a
MATERIAL point of the object: re-orienting a frame moves the pivot with the object,
where a posed-space union AABB slides to a different material location (a union AABB is
not rotation-equivariant once there is more than one part). The transport is not here —
it is `rig.lie.pose_apply(P, c_canon)`, the placement acting on a point.

TAKES MATRICES, NOT JOINT DEFINITIONS. FK already exists once, in
`rig.transforms.forward_kinematics`, with the limit clamping, parent-chain
accumulation and cycle check. The pivot must agree with what the renderer DRAWS, so it
uses the renderer's own FK rather than a second copy of the joint arithmetic. Callers
run inside Blender and pass the result down as plain numpy. Use an IDENTITY base to get
canonical-frame transforms: FK conjugates as `base @ A @ base^-1`, which collapses to
`A`. (core/ also cannot import mathutils — see `core/__init__.py`.)

TAKES POINTS, NOT PARTS. A global AABB is a min/max over the union of every point;
per-part grouping exists only because articulation is per-part. Prefer true MESH
VERTICES over a part's 8 `bound_box` corners: `bound_box` is the part's LOCAL
axis-aligned box, so transforming its corners over-bounds the geometry whenever the
part's transform carries rotation — which an open joint is by definition (measured
0.0066 of centre error at 45 degrees, object longest dim ~= 1). The pivot is computed
once per run, so vertices cost one matmul.
"""

import numpy as np

# What produced a pivot, recorded on the wire so a FUTURE change to the definition is
# detectable rather than silent (the previous definition was never serialized, which is
# why changing it could re-interpret every archived order with nothing on disk to show
# it). See `pivot_record`.
SOURCE_FK_AABB = "fk_aabb"          # this module: canonical FK -> AABB -> carry
SOURCE_CANONICAL_ORIGIN = "canonical_origin"   # the object-frame grammar: no pivot


def _as_points(points):
    """A (N,3) float array from a point list/array; rejects the wrong shape LOUDLY.

    A silently-misread point cloud yields a plausible-looking centre, so shape is
    checked here rather than left to broadcast into something meaningless."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(
            f"expected an (N, 3) array of canonical points, got shape {p.shape}. "
            "Pass mesh vertices (preferred) or bbox corners in the CANONICAL frame.")
    if p.shape[0] == 0:
        raise ValueError("expected at least one canonical point, got an empty array")
    return p


def apply_transform(points, M):
    """(N,3) points through a 4x4 homogeneous transform -> (N,3). None is identity.

    Point-CLOUD transform, which is why it is not `rig.lie` — that module's pose
    algebra acts on single points (`pose_apply`) and composes Sim(3) elements; this is
    the same arithmetic vectorized over N rows, and `M` here is a joint transform, not
    a placement. None means unarticulated (a part under no joint passes through)."""
    p = _as_points(points)
    if M is None:
        return p
    M = np.asarray(M, dtype=float)
    if M.shape != (4, 4):
        raise ValueError(f"part transform must be a 4x4 matrix, got {M.shape}")
    return (M[:3, :3] @ p.T).T + M[:3, 3]


def canonical_centre(points_by_part, part_transforms=None):
    """The object's canonical-frame AABB centre AT A GIVEN ARTICULATION.

    points_by_part   — {part_name: (N,3)} canonical points per part (mesh vertices
                       preferred; see the module docstring on why not bbox corners).
                       A bare (N,3) array is accepted for a single unnamed part.
    part_transforms  — {part_name: 4x4} canonical-frame transform to apply to that
                       part's points first (a part absent from the dict, or mapped
                       to None, is unarticulated). Pass the FK output described in
                       the module docstring. None -> the rest configuration.

    Returns (centre, lo, hi) as (3,) float arrays. AABB (not centroid) is deliberate:
    it is the middle of the object's EXTENT, which is what an orbit pivot wants, and it
    matches what the engine computed before, so a single axis-aligned part keeps its
    historical pivot exactly.

    Carry the result into a frame's camera frame with `rig.lie.pose_apply(P, centre)`.

    Raises on empty input: an object with no points has no centre, and returning a
    zero vector would silently pivot at the canonical origin.
    """
    if isinstance(points_by_part, dict):
        items = list(points_by_part.items())
    else:                                    # a bare point array: one unnamed part
        items = [(None, points_by_part)]
    if not items:
        raise ValueError(
            "canonical_centre: no parts supplied, so the object has no centre to "
            "pivot about. A zero vector here would silently orbit the canonical "
            "origin instead.")
    transforms = part_transforms or {}
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name, pts in items:
        world = apply_transform(pts, transforms.get(name))
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
    return (lo + hi) / 2.0, lo, hi


def pivot_record(centre, c_canon, joint_states, source=SOURCE_FK_AABB):
    """The pivot as a serializable dict — the wire record of WHERE the sweep orbited.

    `joint_states` is the articulation the canonical centre was measured at, so a
    reader can reproduce it. `source` names the DEFINITION (see SOURCE_*), which is
    what makes the NEXT change to it detectable.

    Rounding is deliberately NOT applied here: the caller rounds at the serializer with
    `core.state_json.round_component`, so the pivot prints at the same precision as the
    poses it explains (conventions/state_json.md §5).
    """
    centre = np.asarray(centre, dtype=float).reshape(3)
    return {
        "centre": [float(v) for v in centre],
        "depth_z": float(centre[2]),
        "canonical_centre": [float(v) for v in
                             np.asarray(c_canon, dtype=float).reshape(3)],
        "joint_states": {str(k): float(v) for k, v in (joint_states or {}).items()},
        "source": source,
    }
