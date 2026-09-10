"""
transforms.py — pose math for the reconstruction rig (runs INSIDE Blender).

Separation of concerns (this is the point of the module):

  * scene.py is DECLARATIVE. build() creates parts in a CANONICAL object frame
    (+Z up, centered at the origin, longest dimension ~= 1, no camera awareness).
    Each frame's `pose` (R, t) + the shared SCALE state the object's placement in
    that frame's camera-0 frame.
  * The AGENT (or, later, an optimizer over video) is what ESTIMATES those
    numbers. This module never guesses pose — it only COMPOSES and APPLIES it.
  * The harness owns APPLICATION: compose T * R * S, apply it to the canonical
    parts for rendering, and serialize it as an artifact (pose.json).

This same "declare -> compose -> apply" split extends to articulation: an object
pose is a single rigid+scale transform; a joint chain is several of them composed
down a parent/child tree. forward_kinematics() below is LIVE — the render loop
applies it per frame on top of that frame's base placement, at the frame's joint
states. Estimation of both the per-frame poses and joint states stays outside the
harness (the agent now, an optimizer over video later), exactly like object pose.

Multi-frame model: build() + SCALE + the JOINTS definitions are SHARED across
frames (object identity); each frame carries its own base pose and joint_states
(the object is observed under camera motion + manipulation).
"""

import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

from core import joints as joints_core
from rig import fk as fk_core
from rig import lie


# --------------------------------------------------------------------------- #
# numpy <-> mathutils boundary
# --------------------------------------------------------------------------- #
# The pose MATH lives in rig/lie.py (pure numpy, so it is unit-testable outside
# Blender and shared with the analysis env). This module is the Blender-facing
# layer: it converts between numpy 4x4s and mathutils.Matrix at the boundary
# (the same pattern as camera.relative_extrinsic) so the harness keeps handing
# mathutils matrices to bpy while the arithmetic is the tested lie core.
def _np_to_mat(M4):
    """numpy 4x4 -> mathutils.Matrix."""
    return Matrix([[float(v) for v in row] for row in M4])


def _mat_to_np(M):
    """mathutils.Matrix -> numpy 4x4."""
    return np.array([[float(v) for v in row] for row in M], dtype=float)


# --------------------------------------------------------------------------- #
# object pose:  M = T * R * S   (scale, then rotate, then translate)
# --------------------------------------------------------------------------- #
def _scale_matrix(scale):
    """Uniform scalar -> 4x4 scale matrix; anisotropic scale is forbidden."""
    s = lie.scalar_scale(scale)
    return Matrix.Diagonal((s, s, s, 1.0))


def _unit_quat(q):
    """A normalized mathutils.Quaternion from an authored (w, x, y, z).

    Authored quaternions are routinely non-unit at the 1e-6 level: the sweep's
    paste blocks round to six decimals, and hand edits round further. A non-unit
    q makes to_matrix() carry |q|^2 into the placement, which decompose() then
    reads back as spurious (and numerically ANISOTROPIC) scale — crashing
    write_pose_json at the end of an otherwise-successful pass. Normalizing at
    the parse boundary makes every downstream matrix exactly rigid."""
    vals = [float(v) for v in q]
    # normalizing ~1e-6 JSON drift is silent + correct; a norm far from 1 is a
    # MALFORMED quaternion (reads as the wrong rotation once normalized), so warn.
    lie.check_quat_norm(vals, where=f"_unit_quat q={tuple(vals)}")
    q = Quaternion(vals)
    if q.magnitude == 0.0:
        raise ValueError("pose quaternion has zero magnitude")
    return q.normalized()


def rotation_matrix_of(pose):
    """The 3x3 rotation of a per-frame `pose`, from WHICHEVER key it carries.

    A pose may declare its rotation as a scalar-first unit `quaternion`
    (w, x, y, z) — the new default authoring form, and what the sweep pastes — OR
    as legacy `rotation_euler` (+ optional `euler_order`). Existing runs author
    euler; new runs author quaternion; both must load. Quaternion wins if present.
    """
    p = dict(pose or {})
    q = p.get("quaternion")
    if q is not None:
        return _unit_quat(q).to_matrix()
    return Euler(tuple(p.get("rotation_euler", (0.0, 0.0, 0.0))),
                 p.get("euler_order", "XYZ")).to_matrix()


def compose_pose(rotation_euler=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0),
                 scale=1.0, euler_order="XYZ", quaternion=None):
    """Compose a placement's fields into a single world transform.

    quaternion:     scalar-first unit (w, x, y, z); when given it defines R and
                    `rotation_euler`/`euler_order` are ignored (the new default
                    authoring form).
    rotation_euler: (rx, ry, rz) radians, applied in `euler_order` (Blender's
                    default XYZ) — legacy authoring form (existing runs).
    translation:    (tx, ty, tz) in the camera-0 frame (+Z ahead of the camera).
    scale:          uniform scalar. Axis-specific scale is forbidden.

    Returns M such that a canonical point p maps to M @ p = T @ R @ S @ p.
    """
    T = Matrix.Translation(Vector(translation))
    if quaternion is not None:
        R = _unit_quat(quaternion).to_matrix().to_4x4()
    else:
        R = Euler(rotation_euler, euler_order).to_matrix().to_4x4()
    S = _scale_matrix(scale)
    return T @ R @ S


def normalize_placement(placement):
    """Fill a placement dict (rotation/translation/scale) with defaults.

    Used by the pose diagnostics (sweep) that reason about a full
    placement WITH scale. Per-frame `pose` dicts carry no scale (SCALE is
    shared); build a full placement from one via `frame_placement`. A
    `quaternion` key (new authoring form) is carried through when present, so
    compose_pose(**normalize_placement(p)) honors it over `rotation_euler`.

    `scale` is REQUIRED and has no default.

    A missing scale is UNKNOWABLE here — this function sees a placement, not the
    object it belongs to — so the only correct answers are "the caller states it"
    or "raise". `frame_placement` is the seam that knows the shared SCALE; a
    scaleless placement reaching this function is a caller that skipped it.
    """
    p = dict(placement or {})
    if p.get("scale") is None:
        raise ValueError(
            "placement has no 'scale', and there is no safe default: the shared "
            "SCALE is a property of the object, not of a placement. Compose the "
            "frame's placement through transforms.frame_placement(pose, scale), "
            "or state 'scale' explicitly. (A per-frame pose / fragment entry / "
            "paste block deliberately omits scale — conventions/state_json.md §4.)")
    out = {
        "rotation_euler": tuple(p.get("rotation_euler", (0.0, 0.0, 0.0))),
        "translation": tuple(p.get("translation", (0.0, 0.0, 0.0))),
        "scale": lie.scalar_scale(p["scale"]),
        "euler_order": p.get("euler_order", "XYZ"),
    }
    if p.get("quaternion") is not None:
        out["quaternion"] = tuple(float(v) for v in p["quaternion"])
    return out


def compose_placement_with_scale(pose, scale, euler_order=None):
    """Compose a per-frame `pose` (rotation + translation, NO scale) with the
    SHARED object `scale` into a single canonical->camera-0 matrix.

    The `pose` rotation may be a `quaternion` (new) or `rotation_euler` (legacy);
    both are honored. This is the per-frame base transform: every frame reuses
    the one shared scale, so a rigid object cannot 'breathe' across frames —
    scale drift is not even expressible (see AGENT_TASK.md).
    """
    p = dict(pose or {})
    eo = euler_order or p.get("euler_order", "XYZ")
    return compose_pose(p.get("rotation_euler", (0.0, 0.0, 0.0)),
                        p.get("translation", (0.0, 0.0, 0.0)), scale, eo,
                        quaternion=p.get("quaternion"))


def frame_placement(pose, scale):
    """A full placement dict (rotation/translation/scale) from a per-frame pose
    + the shared scale — so the sweep machinery (which thinks in whole
    placements) can be reused unchanged. Per-frame poses carry no scale of
    their own (SCALE is shared), so the explicit `scale` always wins."""
    return normalize_placement({**dict(pose or {}), "scale": scale})


def reseat_pose(pose, rel_matrix, scale):
    """Seed a per-frame pose from the reference pose + a known relative camera
    motion `rel_matrix` (rigid: `inv(c2w[k]) @ c2w[ref]`, carrying the object
    from the reference camera frame into frame k's).

    Returns a NEW pose dict (quaternion + translation, no scale). Because
    rel_matrix is rigid, the shared scale is provably preserved — we compose,
    then read back only the rotation + translation and re-attach nothing (scale
    stays shared). This is the camera-motion analogue of the sweep's pose orders.
    Rotation is read back as a scalar-first `quaternion` (the new authoring form)
    rather than euler — no gimbal/order ambiguity in the seeded pose.
    """
    p = dict(pose or {})
    Mp = rel_matrix @ compose_placement_with_scale(p, scale)
    loc, rot, _scl = Mp.decompose()          # discard scale (SCALE is shared)
    return {
        "quaternion": (rot.w, rot.x, rot.y, rot.z),
        "translation": (loc.x, loc.y, loc.z),
    }


def frame_pose_to_dict(pose, matrix):
    """Serialize a per-frame base pose for pose.json.

    The AUTHORITATIVE rotation is a scalar-first unit `quaternion` (w, x, y, z),
    with a per-frame `scale` echo (the shared SCALE folded into this frame's
    transform). Only these + `translation` are serialized: they fully determine
    the pose.

    `matrix4x4` and `rotation_euler` are NOT serialized: every consumer that needs
    the 4x4 derives it fresh via `lie.object_pose_matrix`, so a stale-matrix bug is
    unrepresentable.

    `matrix` is still consumed here only to DECOMPOSE q/scale (and to validate
    uniform scale), regardless of how the source `pose` authored its rotation."""
    loc, rot, scl = _np_to_mat(matrix).decompose() if not isinstance(
        matrix, Matrix) else matrix.decompose()
    sx, sy, sz = float(scl.x), float(scl.y), float(scl.z)
    # Relative tolerance: mathutils matrices are float32, so decompose() of an
    # exactly-rigid transform still reads back ~5e-7 anisotropy PER UNIT SCALE.
    # An absolute 1e-6 cutoff spuriously rejected legitimate poses at larger
    # scales (and, before rotation_matrix_of normalized authored quaternions,
    # crashed whole passes on six-decimal-rounded quats).
    tol = 1e-5 * max(1.0, abs(sx))
    if abs(sx - sy) >= tol or abs(sx - sz) >= tol:
        raise ValueError(
            "object pose matrix contains anisotropic scale; only one uniform "
            "scalar scale is allowed")
    return {
        "quaternion": [rot.w, rot.x, rot.y, rot.z],   # authoritative rotation
        "translation": [loc.x, loc.y, loc.z],
        "scale": sx,                                  # per-frame echo of shared SCALE
    }


# --------------------------------------------------------------------------- #
# articulation — forward kinematics (LIVE in the render path)
# --------------------------------------------------------------------------- #
# JOINTS is declarative, SHARED across frames (joint DEFINITIONS, no state):
#   {
#     "name":   "door_hinge",
#     "type":   "revolute" | "prismatic" | "fixed",
#     "child":  "door" | ["door", "handle", ...],  # part name OR the rigid GROUP
#                                                   # of parts that ride the joint
#     "origin": (x, y, z),        # joint origin in the canonical frame
#     "axis":   (x, y, z),        # unit joint axis in the canonical frame
#     "limit":  (lo, hi),         # OPTIONAL — the joint's VALID RANGE (revolute:
#                                 # degrees; prismatic: canonical units). rest=0
#                                 # should lie in [lo, hi]. Absent/None = free
#                                 # (a continuous joint, e.g. a spinning wheel).
#                                 # States are CLAMPED to it everywhere the joint
#                                 # is applied, so no impossible config renders /
#                                 # ships (URDF <limit lower= upper=>).
#     "parent": "other_joint",    # OPTIONAL — name of the joint this one hangs
#                                 # off (kinematic chain); absent = attached to
#                                 # the object base (a single hop).
#   }
# The per-frame DOF VALUES live in a separate `joint_states` map {name: state}
# (revolute: degrees; prismatic: canonical units) — one such map per frame — so
# the same shared definitions re-pose to any observed configuration.
def joint_limit(joint):
    """The joint's valid range as a (lo, hi) float tuple, or None if unbounded.

    STRICT (this is the scene-load path): raises ValueError on a malformed
    limit so a typo in scene.py fails loudly rather than silently disabling
    the clamp. See core.joints.joint_limit."""
    return joints_core.joint_limit(joint, strict=True)


# The joint ARITHMETIC (clamp, local transform, parent-chain accumulation)
# lives in rig/fk.py on pure numpy — the same split as rig/lie.py, so the
# analysis env can pose part meshes off-Blender (analysis.self_intersection)
# against the renderer's own FK rather than a second copy of it. These are the
# mathutils-facing delegates, kept so every Blender-side caller and the JOINTS
# schema doc above stay exactly where they were.
def clamp_joint_state(joint, state):
    """Clamp a DOF `state` to the joint's `limit` (if any) — see rig.fk."""
    return fk_core.clamp_joint_state(joint, state)


def joint_transform(joint, state):
    """Local (canonical-frame) transform a joint applies to its group at `state`,
    as a mathutils.Matrix — the arithmetic is rig.fk.joint_transform.

    revolute  -> rotation `state` DEGREES about `axis` through `origin` (the
                 agent-facing unit everywhere: scene.py, specs, reports).
    prismatic -> translation `state` along `axis`.
    fixed     -> identity.
    `state` is clamped to the joint's `limit` as a runtime backstop.
    """
    return _np_to_mat(fk_core.joint_transform(joint, state))


def forward_kinematics(joints, base_pose=None, states=None):
    """Per-PART additional world transform for the JOINTS at the given states,
    as mathutils.Matrix values — the arithmetic is rig.fk.forward_kinematics
    (see there for the full contract, including why a missing state is 0.0 as
    ARITHMETIC while rig.scene.pose_frame REFUSES an incomplete dict).

    `joints`  — the shared joint DEFINITIONS (see schema above).
    `base_pose` — the object's per-frame base placement matrix M_k, so joint
                transforms declared in the canonical frame act in the posed
                frame (conjugation base @ A @ base.inverted()).
    `states`  — {joint_name: value} for THIS frame (missing -> 0.0 -> rest).

    Returns {part_name: extra_world_transform} for every part in any joint's
    child group; parts under no joint are absent (they stay at the base pose).
    Empty `joints` -> {} (no articulation). Supports parent chains.
    """
    base_np = None if base_pose is None else _mat_to_np(base_pose)
    out = fk_core.forward_kinematics(joints, base_np, states)
    return {part: _np_to_mat(W) for part, W in out.items()}
