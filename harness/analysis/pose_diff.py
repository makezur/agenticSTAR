"""pose_diff.py — the residual between two poses of one articulated object.

Our object poses are `(s, R_i, t_i)` with a SHARED scale, acting canonical -> that
frame's camera as `p_cam = s R_i p + t_i` (POSE.md §3). Each frame also carries an
externally-provided CAMERA pose `(R_cam_i, t_cam_i)`, so in world coordinates

    p_world = s (R_cam_i R_i) p + (R_cam_i t_i + t_cam_i)

This module reports the residual between two such frames, and the two halves live
in DIFFERENT frames on purpose — read the tag, not the name:

  * ROTATION is genuinely world. W_i = R_cam_i R_i is a true world orientation, and
    the residual is the BODY-frame increment `W_i^T W_j` (§ROTATION below).
  * TRANSLATION is camera-centred but world-ORIENTED. Under the reparametrisation
    t_i = s tau_i, the only part a re-pose can move is `u_i = R_cam_i tau_i`: world
    axes, object units. The camera TRANSLATION t_cam_i is DROPPED — it lives in the
    tracker's own gauge, which need not agree with the object's shared s, so adding
    it would mix two incommensurable units and then divide the sum by an object
    scale that half of it has nothing to do with. Consequence, stated plainly: a
    static object filmed by a TRANSLATING camera reports non-zero translation. That
    is the honest price of a gauge-free number (a purely ROTATING camera reads 0).

`pose_frame` names which case a result is in: "oriented" (both frames carry a
camera pose — rotation world, translation world-oriented) or "camera" (no camera
pose, so both halves are in camera-k and conflate the camera's own motion).

Joint deltas are frame-invariant and identical in every case.

The bottom of the analysis stack: every analysis.temporal tool builds on this, so
these are the conventions of every temporal number in the repo. Measures and
reports — no thresholds, no verdict. pose_diff.md has the why.

Document INGEST is analysis.lib.pose_read (re-exported here as `load_pose_json` /
`frame_entry`): it owns reading a pose.json frame record and the state_json.md gates
on it, so this module is only the maths.

Pure numpy + stdlib on rig.lie — no bpy / mathutils / scipy, so it runs in BOTH
Blender's bundled python and the `artscript` env. Runs from harness/:
  micromamba run -n artscript python -m analysis.pose_diff \
      --pose-json RUN/mesh/pose.json --frame-a 000000.jpg --frame-b 000080.jpg
"""

import argparse

import json
import math

import numpy as np

from core import joints as joints_core
from analysis.lib import pose_read
from analysis.lib.io import write_json
from rig import lie


# joint limits: tolerant parse shared via bpy-free core.joints
_joint_limit = joints_core.joint_limit


def _joint_defs_by_name(joints):
    """{name: joint_def} for a JOINTS list (empty/None -> {})."""
    return {j.get("name"): j for j in (joints or []) if j.get("name")}


# --------------------------------------------------------------------------- #
# ingestion — a frame's object_pose dict -> lie.Pose
# --------------------------------------------------------------------------- #
def pose_to_lie(object_pose, scale=None):
    """One frame's `object_pose` dict -> a lie.Pose (uniform scale, quaternion, t).

    ALWAYS from the authoritative scalar-first `quaternion` + `translation` (+ the
    object_pose's own `scale`, else the shared `scale`, else 1.0), NEVER from a
    stored `matrix4x4`. The matrix is a derived cache that is not serialized
    (see rig.transforms.frame_pose_to_dict); this defers to the single shared entry
    point lie.pose_from_object_pose, which ignores it.
    """
    return lie.pose_from_object_pose(object_pose, scale)


# INGEST lives in analysis.lib.pose_read, which owns the state_json.md gates (§1
# quaternion+translation present, §2 joints not nested, §8 no defaulted rotation) so
# every reader gets them, not just this one. Re-exported because a residual and the
# read that feeds it are used together at every call site.
load_pose_json = pose_read.load_pose_json
frame_entry = pose_read.frame_entry


# --------------------------------------------------------------------------- #
# joint deltas — THE home for per-joint change math
# --------------------------------------------------------------------------- #
def joint_deltas(states_a, states_b, joints=None):
    """Signed per-joint change between two joint-state maps.

    `states_a`, `states_b` — {name: value}; a joint absent from a map is REST
        (0.0), per JOINTS.md §3. Every joint seen in EITHER map is reported
        (union), sorted by name.
    `joints` — the shared JOINTS definition list (for each joint's type +
        limit); optional (type/limit are then None).

    Returns {name: {delta, state_a, state_b, type, limit}} — `delta` and both states
    in native units (degrees for revolute, canonical units for prismatic), with the
    declared bounds beside them. The single implementation behind pair_residual,
    and the per-joint ladder in analysis.temporal.derivatives.

    NO percent-of-travel field. `delta / (limit[1] - limit[0])` is the fraction of
    declared travel a step consumed, and both operands are right here — so reporting
    it too would be a third name for information already on the record, and a name a
    reader has to look up to be sure which way it is signed. A consumer that wants
    the cross-joint comparison (a revolute's degrees against a prismatic's units)
    forms the ratio and names it in its own terms."""
    sa = states_a or {}
    sb = states_b or {}
    defs = _joint_defs_by_name(joints)
    out = {}
    for name in sorted(set(sa) | set(sb)):
        a = float(sa.get(name, 0.0))
        b = float(sb.get(name, 0.0))
        jd = defs.get(name, {})
        lim = _joint_limit(jd)
        out[name] = {
            "delta": b - a,
            "state_a": a,
            "state_b": b,
            "type": jd.get("type"),
            "limit": list(lim) if lim else None,
        }
    return out


def joint_states(states, joints=None):
    """Each joint's ABSOLUTE state along a track — the per-frame counterpart to
    `joint_deltas`' per-step change.

    {name: {state, type, limit}}: the state in native units (degrees for revolute,
    canonical units for prismatic) with the declared bounds beside it, so a reader
    can see both where the joint is and what it is allowed to be.

    NO percent here, matching `joint_deltas`. Neither "where the joint sits"
    (`(state - lo) / span`) nor "how much travel the step consumed" (`delta / span`)
    is reported: both are one division away from fields on the record, and shipping
    them invited a specific misreading — the two are DIFFERENT quantities that a
    single obvious name (`percent_of_range`) fits equally well, and they can agree by
    coincidence, agree while meaning opposite things (0% = "did not move" vs 0% =
    "pinned at the limit"), or disagree outright. `state` and `limit` are both here,
    so a caller computes the ratio it wants and names it in its own terms."""
    defs = _joint_defs_by_name(joints)
    out = {}
    for name in sorted(states or {}):
        jd = defs.get(name, {})
        lim = _joint_limit(jd)
        out[name] = {"state": float((states or {})[name]),
                     "type": jd.get("type"),
                     "limit": list(lim) if lim else None}
    return out


# --------------------------------------------------------------------------- #
# THE per-frame lift — the seam everything else is built from
# --------------------------------------------------------------------------- #
def camera_rotation(camera_c2w):
    """The ROTATION block R_cam of a 4x4 camera-to-world pose, orthonormalized.

    Only the rotation is taken: the camera's translation is deliberately unused
    everywhere in this module (see the module docstring). The SVD projection to the
    nearest proper rotation mirrors lie.pose_from_matrix, so an externally-provided
    camera matrix that is off-orthonormal by fp drift cannot leak a non-rotation
    (or a reflection) into a residual."""
    R = np.asarray(camera_c2w, dtype=float)[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0.0:
        U[:, -1] *= -1.0
        R_ortho = U @ Vt
    return R_ortho


def lift_placement(object_pose, camera_c2w, scale):
    """One frame -> `(R_world, offset_canon)`, the two gauge-free placement parts.

    `R_world` = `R_cam @ R_obj` — a true WORLD orientation of the object.
    `offset_canon` = `R_cam @ t_obj / s` = `R_cam tau` — the camera->object vector
        in WORLD AXES and OBJECT UNITS (fractions of the object's longest canonical
        dimension).

    This is the primitive the whole temporal stack is built from, and the first
    PER-FRAME world quantity in the repo: pair residuals difference two of these,
    and analysis.temporal.derivatives differences them twice (velocity, then
    acceleration), which a pairwise-only API cannot express.

    Needs only the camera's ROTATION. The camera translation is dropped because
    (a) it is in the tracker's gauge, not the object's, and (b) it is the one term
    no re-pose of the object can change — so excluding it makes the number depend
    on exactly what a refiner controls. See the module docstring for the visible
    consequence.

    `scale` — the shared object SCALE; <= 0 is treated as 1.0.
    """
    raw_scale = lie.scalar_scale(scale)
    s = raw_scale if raw_scale > 0 else 1.0
    pose = pose_to_lie(object_pose, s)
    R_cam = camera_rotation(camera_c2w)
    R_world = R_cam @ lie.quat_to_matrix(pose.q)
    offset_canon = R_cam @ (np.asarray(pose.t, dtype=float) / s)
    return R_world, offset_canon


def _rotation_block(R_rel):
    """A relative rotation matrix -> {rotation_deg, rotation_axis}.

    The geodesic ANGLE in [0, 180] plus the unit axis it turns about; the axis is
    None for a ~0 rotation, which genuinely has no direction (never a fabricated
    zero vector). Axis-times-angle as a single 3-vector is deliberately NOT
    reported: it is the same information multiplied together, and in degrees it is
    not a usable Lie-algebra element, so it was a field that could only ever
    disagree with these two."""
    rotvec = lie.matrix_to_rotvec(R_rel)
    angle = float(np.linalg.norm(rotvec))
    return {
        "rotation_deg": math.degrees(angle),
        "rotation_axis": (rotvec / angle).tolist() if angle > 1e-9 else None,
    }


# --------------------------------------------------------------------------- #
# THE residual convention — one implementation, both frames of reference
# --------------------------------------------------------------------------- #
def pair_residual(a, b, scale, joints=None):
    """THE residual from frame a to frame b. One shape, whatever the inputs.

    `a`, `b` — each `(object_pose, joint_states, camera_c2w)` for one frame.
        `camera_c2w` may be None on BOTH (then `pose_frame` is "camera"); exactly
        one being None raises, since the two frames would not share a frame of
        reference.
    `scale` — the shared object SCALE (<= 0 is treated as 1.0).
    `joints` — the shared JOINTS definition list (types + limits); optional.

    Returns {pose_frame, pose: {...}, joints: {...}}, where `pose` is

      rotation_deg, rotation_axis   the geodesic angle of `W_i^T W_j` + its axis
      translation_canon (3)         u_j - u_i, world axes, object units
      translation_dist_canon        |u_j - u_i|
      translation_dist              s * |u_j - u_i|, scene units (metric, so it is
                                    comparable with a depth residual, which is
                                    measured in the same units)
      offset_canon_a/_b (3)         each frame's own u — reported because the
                                    velocity/acceleration ladder in
                                    analysis.temporal.derivatives needs the states,
                                    not just their difference

    ROTATION is the BODY-frame increment `W_i^T W_j` (an axis in frame i's own
    object frame), NOT the spatial `W_j W_i^T`. The angle is identical either way
    (the two are conjugate), so only the axis is at stake. Body is chosen because
    it is what an increment applied TO frame i's pose is expressed in, and body
    increments telescope right-to-left: (W_i^T W_j)(W_j^T W_k) = W_i^T W_k. The
    caveat that comes with it: axes at different i live in DIFFERENT frames, so two
    steps' axes are not comparable with each other.

    TRANSLATION is a plain difference of two `u`, which share world axes, so
    consecutive deltas do add. The shared SCALE enters once, inside `lift_placement`.

    Which frame `pose` lives in is on the result as `pose_frame`, and the two halves
    are NOT in the same frame in the "oriented" case — see the module docstring.
    """
    raw_scale = lie.scalar_scale(scale)
    s = raw_scale if raw_scale > 0 else 1.0
    op_a, st_a, cam_a = a
    op_b, st_b, cam_b = b

    if (cam_a is None) != (cam_b is None):
        # ONE camera pose is an error, not a fallback. Falling back to the camera
        # frame would throw away the camera we do have and silently relabel the
        # result; using the one we have for both frames would invent a camera the
        # run never observed. Either way the number would be a quiet mix of two
        # conventions, which is exactly what `pose_frame` exists to prevent.
        missing = "a" if cam_a is None else "b"
        raise ValueError(
            f"frame {missing} has no camera pose while the other does; a residual "
            "needs BOTH frames in the same frame of reference. Supply both camera "
            "poses (-> pose_frame 'oriented') or neither (-> 'camera').")
    # no camera pose on either frame: lift with R_cam = I, which leaves the object
    # rotation/translation in camera-k where they already are. Same algebra either
    # way, so there is ONE code path and the only difference is what was composed.
    eye = np.eye(4)
    pa = lift_placement(op_a, cam_a if cam_a is not None else eye, s)
    pb = lift_placement(op_b, cam_b if cam_b is not None else eye, s)
    residual = placement_residual(pa, pb, s)
    return {
        "pose_frame": "oriented" if cam_a is not None else "camera",
        "pose": residual,
        "joints": joint_deltas(st_a, st_b, joints),
    }


def placement_residual(a, b, scale):
    """THE residual between two LIFTED placements — the single implementation.

    `a`, `b` — each `(W, u)` from `lift_placement`: a world orientation and a
        camera->object offset in object units.
    `scale` — the shared object SCALE, for the metric translation only.

    Returns the `pose` block of §2: rotation (body-frame `W_a^T W_b`, as a geodesic
    angle + unit axis), `translation_canon` = `u_b - u_a` with its two magnitudes,
    and each placement's own offset.

    Kept separate from `pair_residual` because there are two ways to arrive at a
    pair of placements and only one residual: `pair_residual` lifts two frame
    records, while analysis.temporal.sequence lifts a whole track ONCE and pairs
    consecutive rows off it (an N-frame sequence would otherwise lift every frame
    three times). Both call this, so a step residual has exactly one definition and
    the two paths cannot drift — which is also why the derivative layer reads the
    step records instead of re-differencing the track: a first difference of
    consecutive placements IS this residual, so computing it twice would be two
    implementations of one quantity.
    """
    raw_scale = lie.scalar_scale(scale)
    s = raw_scale if raw_scale > 0 else 1.0
    (Wa, ua), (Wb, ub) = a, b
    du = np.asarray(ub, dtype=float) - np.asarray(ua, dtype=float)
    dist = float(np.linalg.norm(du))
    omega = np.asarray(Wa, dtype=float).T @ np.asarray(Wb, dtype=float)
    return {
        **_rotation_block(omega),
        "translation_canon": [float(v) for v in du],
        "translation_dist_canon": dist,
        "translation_dist": float(s * dist),
        "offset_canon_a": [float(v) for v in ua],
        "offset_canon_b": [float(v) for v in ub],
        # the rotation INCREMENT as a matrix, for a consumer that composes it
        # (analysis.temporal.derivatives' angular acceleration). Numpy, not JSON, and
        # stripped before the report ships — same contract as frame_track's
        # `placement`. It rides along so nobody rebuilds Omega from the reported
        # axis+angle: those are DEGREES, a reporting unit, and the round trip
        # matrix -> axis+deg -> radians -> matrix is both lossy and pointless when
        # the matrix is right here.
        "rotation_increment": omega,
    }


# --------------------------------------------------------------------------- #
# programmatic entry points — the "easy to run on real scenes" surface
# --------------------------------------------------------------------------- #
def diff_frames(pose_json, frame_a, frame_b):
    """The primary programmatic entry: the residual between two frames.

    `pose_json` — a loaded pose.json dict or a path to one. The shared `scale` and
    `joint_defs` are pulled from it automatically, so a script needs only the two
    frame names. Returns
    {"frame_a", "frame_b", "scale", "pose_frame", "pose", "joints"} — `pair_residual`'s
    blocks with the two frame names and the scale they were measured against.

    `pose_frame` is "oriented" when BOTH frames carry a camera pose and "camera"
    when NEITHER does; the two mean different things and must never be compared
    (module docstring). Exactly one camera pose raises — see `pair_residual`.
    """
    pj = load_pose_json(pose_json)
    scale = lie.scalar_scale(pj.get("scale", 1.0), "pose.json scale")
    joints = pj.get("joint_defs") or pj.get("JOINTS") or []
    entry_a = frame_entry(pj, frame_a)
    entry_b = frame_entry(pj, frame_b)
    residual = pair_residual(entry_a, entry_b, scale, joints)
    return {"frame_a": frame_a, "frame_b": frame_b, "scale": scale, **residual}


def frame_track(pose_json, order):
    """Per-frame placement states along `order` — lifting each frame EXACTLY ONCE.

    One row per frame: {frame, offset_canon, orientation_world_deg, joints,
    placement}, where `orientation_world_deg` is the frame's ABSOLUTE world
    orientation as an axis-angle 3-vector in degrees (`log(W_i)`) and `joints` is
    `joint_states` (where each joint SITS in its travel).

    `placement` is the raw `(W, u)` the row was built from — numpy, not JSON, and
    dropped before serialization by analysis.temporal.sequence. It is here so a
    caller pairing consecutive rows into step residuals uses the MATRIX rather than
    re-exponentiating `orientation_world_deg`: that field is rounded to a reporting
    unit (degrees), so a round trip through it would make a step residual that
    disagrees with `pair_residual` in the last digits for no reason.

    A frame with no camera pose is lifted with `R_cam = I`, which leaves the pose in
    camera-k where it already is — so `placement` is ALWAYS present and a caller can
    always pair rows (the camera-frame branch of `pair_residual` does the same
    thing). What such a frame does NOT get is `offset_canon` /
    `orientation_world_deg`: those name WORLD quantities, and without a camera there
    is no world to name, so they are None rather than a camera-frame number wearing a
    world label. That is also what makes a camera-less frame visible as a gap in the
    derivative layer, which reads those fields.

    This is the primitive the whole temporal stack reads: step residuals pair
    consecutive rows (analysis.temporal.sequence), and the derivative ladder
    differences them (analysis.temporal.derivatives). Both off ONE lift per frame.
    """
    pj = load_pose_json(pose_json)
    scale = lie.scalar_scale(pj.get("scale", 1.0), "pose.json scale")
    joints = pj.get("joint_defs") or pj.get("JOINTS") or []
    eye = np.eye(4)
    rows = []
    for name in order:
        op, states, cam = frame_entry(pj, name)
        placement = lift_placement(op, cam if cam is not None else eye, scale)
        W, u = placement
        rows.append({
            "frame": name,
            # world-named fields only when a camera made them world (see docstring)
            "offset_canon": None if cam is None else [float(v) for v in u],
            "orientation_world_deg": None if cam is None else [
                math.degrees(float(v)) for v in lie.matrix_to_rotvec(W)],
            "joints": joint_states(states, joints),
            "placement": placement,
        })
    return rows


# --------------------------------------------------------------------------- #
# human-readable summary
# --------------------------------------------------------------------------- #
_FRAME_NOTES = {
    "oriented": ("rotation in world frame; translation camera-centred, world axes "
                 "(camera translation excluded — a translating camera shows up)"),
    "camera": "camera frame (includes the camera's own motion — no camera pose)",
}


def _summary_lines(result):
    """Compact human lines for one diff_frames result."""
    p = result["pose"]
    frame = result.get("pose_frame", "camera")
    lines = [
        f"{result['frame_a']} -> {result['frame_b']}  "
        f"(scale {result['scale']}, {_FRAME_NOTES.get(frame, frame)})",
        f"  translation: {p['translation_dist_canon']:.4f} of object size "
        f"= {p['translation_dist']:.4f} scene units "
        f"(canon delta {[round(v, 4) for v in p['translation_canon']]})",
        f"  rotation:    {p['rotation_deg']:.2f} deg  (~180 = a full flip)",
    ]
    if result["joints"]:
        lines.append("  joints:")
        for name, j in result["joints"].items():
            # revolute deltas are DEGREES (say so; degree magnitudes don't need
            # 4 decimals) — prismatic stay canonical units at full precision.
            delta = (f"{j['delta']:+.1f} deg" if j.get("type") == "revolute"
                     else f"{j['delta']:+.4f}")
            lim, d = j.get("limit"), j.get("delta")
            span = (float(lim[1]) - float(lim[0])) if lim else 0.0
            note = (f"{100.0 * float(d) / span:+.1f}% of range"
                    if lim and abs(span) > 1e-12 else "n/a (no limit)")
            lines.append(f"    {name:>16}: delta {delta}  {note}")
    return lines


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="compare two poses of the same articulated object: the "
                    "base-pose residual (rotation as a geodesic angle in degrees, "
                    "translation as a fraction of object size) and each joint's "
                    "change as a percent of its declared range. With a camera pose "
                    "on both frames the rotation is in world frame and the "
                    "translation is world-oriented about each camera (the camera's "
                    "own translation is excluded); without one both are in camera "
                    "frame. Every result is tagged pose_frame. For per-step "
                    "residuals along the whole frame sequence use "
                    "analysis.temporal.sequence")
    p.add_argument("--pose-json", default="", help="a scene's pose.json; "
                   "--frame-a/--frame-b pick the pair to diff")
    p.add_argument("--frame-a", default="", help="first frame name (pose.json mode)")
    p.add_argument("--frame-b", default="", help="second frame name (pose.json mode)")

    g = p.add_argument_group("ad-hoc mode (inline poses, no pose.json)")
    g.add_argument("--pose-a", default="", help="inline JSON object_pose A "
                   "({quaternion, translation, joints})")
    g.add_argument("--pose-b", default="", help="inline JSON object_pose B")
    g.add_argument("--joints", default="", help="inline JSON JOINTS defs list "
                   "(for limits/types)")
    g.add_argument("--scale", type=float, default=1.0, help="shared object SCALE "
                   "(ad-hoc mode; pose.json mode reads it from the file)")

    p.add_argument("--out", default="", help="write the JSON report here")
    args = p.parse_args()

    if args.pose_json:
        if not (args.frame_a and args.frame_b):
            p.error("--pose-json needs --frame-a and --frame-b")
        report = diff_frames(args.pose_json, args.frame_a, args.frame_b)
    elif args.pose_a and args.pose_b:
        op_a = json.loads(args.pose_a)
        op_b = json.loads(args.pose_b)
        joints = json.loads(args.joints) if args.joints else []
        st_a = op_a.pop("joints", {})
        st_b = op_b.pop("joints", {})
        residual = pair_residual((op_a, st_a, None), (op_b, st_b, None),
                                 args.scale, joints)
        report = {"frame_a": "A", "frame_b": "B", "scale": args.scale, **residual}
    else:
        p.error("give --pose-json + --frame-a/--frame-b, "
                "or --pose-a and --pose-b")

    print("\n".join(_summary_lines(report)))
    if args.out:
        # `rotation_increment` is a numpy matrix for in-process consumers
        # (temporal.derivatives); it is not part of the JSON report.
        (report.get("pose") or {}).pop("rotation_increment", None)
        report.pop("rotation_increment", None)
        write_json(args.out, report)
        print(f"[pose_diff] wrote {args.out}")


if __name__ == "__main__":
    main()
