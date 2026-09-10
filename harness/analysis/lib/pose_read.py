"""pose_read — the one INGEST seam for a pose.json document, and its gates.

Every analysis reader of a whole-run pose document comes through here: the temporal
stack (sequence, report), and multiagent.windows for its committed
lookups. Reading is a separate job from the residual maths in analysis.pose_diff —
this module answers "what does this frame's record say", the other answers "how far
apart are two of them" — and giving the read its own home is what lets the
conventions/state_json.md gates be enforced ONCE for every caller instead of at each
call site (they were previously enforced at none).

What is checked, and the section it comes from:

  * §1 — `quaternion` and `translation` must both be present. §8: "a missing
    rotation is an error, never identity." A committed record with no rotation means
    the pose is MISSING, and identity is the worst possible substitute because it is
    a plausible-looking pose: the residual comes out large and legible rather than
    failing, so a frame that lost its pose reads as a frame that rotated.
  * §2 — `joints` is a SIBLING of the pose, never nested inside it. A nested
    `joints` silently reads as NO joints, i.e. every joint at rest, which §2 calls
    out as "straightening" the object; the object then differs from the record in a
    way no number reports.
  * §1 — the derived caches `matrix4x4` / `rotation_euler` / `euler_order` must not
    be written. They are ignored structurally (lie.pose_from_object_pose reads only
    the authoritative fields), so their presence is a stale WRITER, not a bad read:
    reported via `warnings.warn`, never fatal, since old files on disk legitimately
    still carry them.

Deliberately NOT checked here: the quaternion's norm (lie.check_quat_norm warns and
Pose renormalizes — §5 says 6dp leaves a unit quaternion non-unit at ~1e-6, handled
at the read boundary, and writers must not compensate) and the scale's shape
(lie.scalar_scale rejects sequence/anisotropic scale wherever a scale is resolved).
Both already have exactly one owner; duplicating them here would give them two.

Pure stdlib + json; no numpy, no bpy.
"""

import json
import warnings

# conventions/state_json.md §1: derived caches that must no longer be written.
DERIVED_CACHE_KEYS = ("matrix4x4", "rotation_euler", "euler_order")


def load_pose_json(pose_json):
    """Accept a loaded pose.json dict OR a path to one; return the dict."""
    if isinstance(pose_json, str):
        with open(pose_json) as f:
            return json.load(f)
    return pose_json


def _frames_of(pose_json):
    """The document's frame map, under either spelling."""
    return pose_json.get("frames") or pose_json.get("FRAMES") or {}


def frame_entry(pose_json, name, require_pose=True):
    """(object_pose, joints, camera_c2w) for a frame name in a loaded pose.json.

    Tolerates both the serialized key `object_pose` and a bare `pose` (the scene.py
    authoring key), so the tool ingests either shape — permitted by state_json.md §3
    for a READER as long as it says why, and the why is that a residual is valid on
    either: committed-vs-committed and committed-vs-proposed are both meaningful.

    `camera_c2w` is the frame's 4x4 camera-to-world pose — provided externally, from
    whatever tracker the run used — or None when the run does not know its cameras.
    Only its ROTATION is ever consumed (pose_diff.lift_placement).

    `require_pose` (default True) enforces state_json.md §8: a frame whose record
    lacks `quaternion` or `translation` RAISES. Pass False only where a partial pose
    is a legitimate intermediate — a hand-authored scene.py frame, or an in-progress
    refiner fragment — and then handle the missing fields explicitly rather than
    letting lie.pose_from_object_pose default the rotation to identity.
    """
    frames = _frames_of(pose_json)
    if name not in frames:
        raise ValueError(
            f"frame {name!r} not in pose.json (have: {', '.join(frames)})")
    entry = frames[name] or {}
    op = entry.get("object_pose") or entry.get("pose") or {}

    # §2: joints are a SIBLING of the pose. Nested, they would read as "no joints"
    # = every joint at rest, which silently straightens the object.
    if isinstance(op, dict) and "joints" in op:
        raise ValueError(
            f"frame {name!r} nests `joints` INSIDE its pose; joints are a sibling "
            "of the pose, never nested (state_json.md §2). Nested, they read as no "
            "joints at all — every joint silently at rest.")

    if require_pose:
        missing = [k for k in ("quaternion", "translation") if op.get(k) is None]
        if missing:
            raise ValueError(
                f"frame {name!r} has no pose {' or '.join(missing)}; a missing "
                "rotation/translation is an error, never identity "
                "(state_json.md §8). Identity is a plausible-looking pose, so the "
                "residual would come out large and legible instead of failing.")

    stale = [k for k in DERIVED_CACHE_KEYS if k in op]
    if stale:
        # a stale WRITER, not a bad read: these are ignored structurally, so this
        # cannot change any number — it can only stop the next writer re-adding them.
        warnings.warn(
            f"frame {name!r} carries derived cache key(s) {', '.join(stale)}, which "
            "must no longer be written (state_json.md §1). They are ignored here; "
            "the authoritative quaternion/translation always win.",
            stacklevel=2)

    return op, (entry.get("joints") or {}), entry.get("camera_c2w")


def frame_names(pose_json):
    """The document's frame names, in whatever order it stored them (unordered —
    analysis.frames.order_frames puts them in TEMPORAL order)."""
    return list(_frames_of(pose_json))
