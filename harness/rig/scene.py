"""scene.py — clear the scene, exec the agent's scene.py, pose + serialize.

Owns the object lifecycle: wipe the factory scene, run the agent-authored
build() into the 'parts' collection, and (per frame) pose the canonical parts in
front of a camera via that frame's base pose + joint states. Parts are posed by
transforming matrix_world only (mesh data stays canonical), so the scene can be
restored to canonical before exporting a rest-pose GLB.

MULTI-FRAME + ARTICULATION. The agent's scene.py declares:
  * build()          — parts in the CANONICAL frame (+Z up, centered, ~1 unit),
                       the UNION of parts across all observed frames.
  * SCALE            — the object's SHARED physical size (frames never re-scale).
  * REFERENCE_FRAME  — the frame whose pose anchors the gauge (authored directly).
  * JOINTS           — SHARED joint DEFINITIONS (type/child/origin/axis/parent).
  * FRAMES           — per-frame {pose?, joints?, moved?}: the object's base pose
                       and per-joint DOF values in that frame.
"""

import json
import os
import runpy
from dataclasses import dataclass, field

import bpy
from mathutils import Matrix

from core import joints as joints_core
from core.scene_contract import validate_scene_contract_file
from rig import lie, transforms


@dataclass
class SceneSpec:
    """The declarative scene, normalized. build() has already run."""
    scale: float = 1.0                     # uniform scalar, SHARED
    reference_frame: str = None            # gauge anchor (a key of `frames`)
    joint_defs: list = field(default_factory=list)   # SHARED joint definitions
    frames: dict = field(default_factory=dict)       # name -> normalized entry

    @property
    def frame_names(self):
        return list(self.frames.keys())


def clear():
    """Remove everything so only the agent's geometry (+ our rig) exists."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras,
                 bpy.data.lights):
        for block in list(coll):
            coll.remove(block)


def ensure_parts_collection():
    """Return the 'parts' collection, creating and linking it if needed."""
    parts = bpy.data.collections.get("parts")
    if parts is None:
        parts = bpy.data.collections.new("parts")
        bpy.context.scene.collection.children.link(parts)
    return parts


def mesh_parts():
    """Sorted list of mesh objects in the 'parts' collection."""
    parts = ensure_parts_collection()
    return sorted((o for o in parts.objects if o.type == "MESH"),
                  key=lambda o: o.name)


def _normalize_frame(entry):
    """A single FRAMES value -> {'pose', 'joints', 'moved'} with defaults."""
    e = dict(entry or {})
    pose = e.get("pose")
    return {
        "pose": dict(pose) if pose is not None else None,
        "joints": dict(e.get("joints", {}) or {}),
        "moved": bool(e.get("moved", False)),
    }


def _check_frame_joints(frames, joint_defs):
    """Reject any per-frame joint state that falls outside its joint's `limit`.

    The `limit` and the per-frame `joints` state are authored in the SAME
    scene.py, so a state outside the range is the scene contradicting itself, not
    external noise. We FAIL LOUDLY (rather than silently clamp) so the authored
    numbers, the rendered pixels, and pose.json's recorded state stay identical —
    a recorded state that differs from what was authored is a landmine for the
    downstream URDF. The one-line fix is named in the error (widen the limit, or
    fix the state). All offenders across all frames are collected into one error
    so the agent sees every conflict at once. (joint_transform still clamps at
    render time as a defensive backstop for PROGRAMMATIC drivers — e.g. a future
    video optimizer — whose states never pass through this authored-scene gate.)
    """
    by_name = {j.get("name"): j for j in joint_defs if j.get("name")}
    bad = []
    for fname, entry in frames.items():
        states = entry.get("joints") or {}
        for jname, val in states.items():
            j = by_name.get(jname)
            if j is None:
                continue
            lim = transforms.joint_limit(j)
            if lim is None:
                continue
            lo, hi = lim
            if float(val) < lo - 1e-9 or float(val) > hi + 1e-9:
                bad.append(f"  frame {fname!r}: joint {jname!r} state "
                           f"{float(val):.4g} outside its limit [{lo:.4g}, "
                           f"{hi:.4g}]")
    if bad:
        raise RuntimeError(
            "scene.py declares joint states outside their joint `limit` "
            "(impossible configurations):\n" + "\n".join(bad) + "\n"
            "Fix each frame's joint state, or widen the joint's `limit` in "
            "JOINTS if that configuration is real.")


def _check_frame_joints_complete(frames, joint_defs):
    """Reject any frame that omits a state for a joint the scene articulates.

    The companion to _check_frame_joints, and loud for the same reason: a joint
    state is ABSOLUTE, so an omitted one has no defaultable value (see
    core.joints.require_complete_states). Left to a 0.0 fallback the frame
    renders STRAIGHTENED — a plausible picture of a configuration the scene never
    declared, which then ships into pose.json and the downstream URDF.

    Every offender across every frame is collected into one error, matching
    _check_frame_joints, so an agent that forgot a joint in ten frames sees all
    ten instead of fixing them one render at a time."""
    names = joints_core.articulated_names(joint_defs)
    if not names:
        return
    bad = []
    for fname, entry in frames.items():
        states = entry.get("joints") or {}
        missing = [n for n in names if n not in states]
        if missing:
            bad.append(f"  frame {fname!r}: missing "
                       f"{', '.join(repr(m) for m in missing)}")
    if bad:
        raise RuntimeError(
            "scene.py declares articulated JOINTS but some frames do not give "
            "every joint a state:\n" + "\n".join(bad) + "\n"
            "Joint states are ABSOLUTE, not increments: there is no value that "
            "means 'leave this joint as it was', so an omitted state cannot be "
            "defaulted (0.0 would straighten the joint and render the wrong "
            "configuration). Give every frame a state for every articulated "
            "joint — write the rest state explicitly if that is what you mean.")


def build_from(scene_path):
    """Exec the agent's scene.py and call build(); geometry lands in 'parts'.

    Returns a normalized SceneSpec. build() creates parts in the CANONICAL
    frame; posing happens per frame later via pose_frame().
    """
    validate_scene_contract_file(scene_path)
    ensure_parts_collection()
    before = set(bpy.data.objects)
    ns = runpy.run_path(scene_path)  # module namespace after top-level exec
    scale = lie.scalar_scale(ns.get("SCALE", 1.0), f"{scene_path} SCALE")
    build = ns.get("build")
    if not callable(build):
        raise RuntimeError(f"{scene_path} must define a build() function")
    build()

    # Flush build()'s transforms into matrix_world BEFORE anyone snapshots them.
    # Assigning obj.rotation_euler / .location / .scale only marks the object
    # dirty — Blender does NOT recompute matrix_world until the next depsgraph
    # evaluation (view_layer.update(), or the side effect of an operator like
    # transform_apply). capture_canonical() reads matrix_world directly, so
    # WITHOUT this update a bare `obj.rotation_euler = (...)` at the end of
    # build() (no operator after it) is captured as IDENTITY and silently
    # dropped from the canonical pose — the part renders unrotated in every view
    # (match/turntable/depth alike). This bit the notebook spine cylinder: it was
    # authored to lie along +Y but, built last with no trailing operator, got
    # captured unrotated and skewered the cover perpendicular in every render.
    # box()'s transform_apply incidentally flushed earlier parts, which is why
    # the bug was intermittent (only the last-touched, operator-free part broke).
    bpy.context.view_layer.update()

    parts = ensure_parts_collection()
    for obj in bpy.data.objects:
        if obj not in before and obj.type == "MESH":
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            parts.objects.link(obj)

    joint_defs = list(ns.get("JOINTS", []) or [])
    # Validate each joint's optional `limit` up front (a malformed limit raises
    # here, not deep in the render loop) — see transforms.joint_limit.
    for j in joint_defs:
        transforms.joint_limit(j)
    raw_frames = ns.get("FRAMES")
    if not raw_frames:
        # An old-framing scene defined a single PLACEMENT dict instead of FRAMES.
        # That model is gone — fail loudly rather than silently rendering at the
        # origin (a PLACEMENT is never read; the object would land at identity).
        if ns.get("PLACEMENT") is not None:
            raise RuntimeError(
                f"{scene_path}: PLACEMENT is obsolete — the scene must declare a "
                "FRAMES dict (per-frame {pose, joints, moved}) plus the shared "
                "SCALE / REFERENCE_FRAME / JOINTS. Move PLACEMENT's rotation_euler "
                "+ translation into FRAMES[REFERENCE_FRAME]['pose'] and its scale "
                "into the top-level SCALE. See examples/scene_example.py.")
        # A scene with no explicit frames = a single frame at the origin pose.
        # Stated HERE, as a declaration, rather than left to the reference-pose
        # fill below: this is the one case where identity-at-the-origin is the
        # documented meaning of the scene rather than a guess standing in for a
        # measurement. With it written out, a missing reference pose in an AUTHORED
        # FRAMES dict is unambiguously an error.
        raw_frames = {"frame0": {"pose": {"rotation_euler": (0.0, 0.0, 0.0),
                                          "translation": (0.0, 0.0, 0.0)}}}
    frames = {name: _normalize_frame(entry) for name, entry in raw_frames.items()}
    _check_frame_joints(frames, joint_defs)
    _check_frame_joints_complete(frames, joint_defs)

    reference_frame = ns.get("REFERENCE_FRAME") or next(iter(frames))
    if reference_frame not in frames:
        raise RuntimeError(
            f"REFERENCE_FRAME {reference_frame!r} is not a key of FRAMES "
            f"({', '.join(frames)})")
    # The reference frame anchors the gauge every other frame is seeded FROM
    # (render_wrapper.resolve_frames reseats each seeded frame onto it), so it
    # must carry an AUTHORED pose: a defaulted identity placement is a whole
    # ABSOLUTE placement conjured from nothing, the same class of defect as
    # defaulting a joint state. Refuse it, exactly as render_wrapper:172 refuses a
    # moved frame with no pose.
    if frames[reference_frame]["pose"] is None:
        raise RuntimeError(
            f"REFERENCE_FRAME {reference_frame!r} carries no 'pose'. The "
            "reference frame anchors the gauge that every other frame is seeded "
            "from, so it cannot be defaulted — an identity rotation at the "
            "origin is a guess that silently becomes the run's ground truth. "
            "Author FRAMES[REFERENCE_FRAME]['pose'] (quaternion or "
            "rotation_euler + translation), or point REFERENCE_FRAME at a frame "
            "that has one.")
    return SceneSpec(scale=scale, reference_frame=reference_frame,
                     joint_defs=joint_defs, frames=frames)


# --------------------------------------------------------------------------- #
# posing — one per-frame base pose + joint states, applied to canonical parts
# --------------------------------------------------------------------------- #
def capture_canonical(objs):
    """Snapshot the canonical (rest, unposed) world matrices of the parts.

    build() leaves each part at its canonical transform, so calling this right
    after build_from() records the rest pose. Every frame re-poses FROM this
    (M_k @ canonical[o]), so posing never accumulates."""
    return {o: o.matrix_world.copy() for o in objs}


def restore_world(objs, saved):
    """Set each part back to a snapshotted world matrix (e.g. canonical)."""
    for o in objs:
        if o in saved:
            o.matrix_world = saved[o]


def pose_frame(objs, canonical, pose, scale, joint_defs, joint_states, where=None):
    """Pose the canonical parts for one frame: base placement M_k = T·R·S (shared
    scale) applied to every part, then per-joint forward kinematics applied on
    top for the parts that ride a joint at this frame's states.

    `joint_states` must name EVERY articulated joint. This is the seam every
    render path funnels through, and it is checked here rather than inside
    forward_kinematics because the joint DEFINITIONS are in hand here (FK's
    arithmetic stays total for programmatic drivers — see
    core.joints.require_complete_states and transforms.forward_kinematics).
    `where` names the caller in the message; it defaults to the view/frame-free
    "pose_frame" when a caller has nothing more specific to say.

    Returns M_k (the canonical->camera-k base matrix)."""
    joints_core.require_complete_states(joint_defs, joint_states,
                                       where or "pose_frame")
    M = transforms.compose_placement_with_scale(pose, scale)
    for o in objs:
        o.matrix_world = M @ canonical[o]
    fk = transforms.forward_kinematics(joint_defs, M, joint_states)
    for o in objs:
        extra = fk.get(o.name)
        if extra is not None:
            o.matrix_world = extra @ o.matrix_world
    return M


# --------------------------------------------------------------------------- #
# serialization — the downstream handoff (canonical rig + per-frame poses)
# --------------------------------------------------------------------------- #
def write_pose_json(out_path, spec, frame_results, parts):
    """Serialize the shared rig + per-frame estimated poses/joint states.

    spec           — the SceneSpec (shared scale / reference / joint defs).
    frame_results  — {frame_name: {matrix, view_index?, camera_c2w?, moved?}}
                     as computed by the render loop; the base pose comes from
                     spec.frames[name]["pose"].
    parts          — sorted part names.
    """
    frames_out = {}
    for name, entry in spec.frames.items():
        res = frame_results.get(name, {})
        # the RESOLVED pose (seeded frames have no pose in the spec) + its matrix
        pose = res.get("pose", entry["pose"])
        matrix = res.get("matrix")
        if matrix is None:
            matrix = transforms.compose_placement_with_scale(pose, spec.scale)
        rec = {
            "moved": entry["moved"],
            "object_pose": transforms.frame_pose_to_dict(pose, matrix),
            "joints": entry["joints"],
        }
        if res.get("view_index") is not None:
            rec["view_index"] = res["view_index"]
        if res.get("camera_c2w") is not None:
            rec["camera_c2w"] = [list(r) for r in res["camera_c2w"]]
        if res.get("intrinsics") is not None:
            # the K actually rendered through, + where it came from (so a
            # downstream reader never has to guess whether it was a real
            # measured K or the iPhone-13 placeholder).
            rec["intrinsics"] = {"provenance": res.get("intr_provenance"),
                                 **res["intrinsics"]}
        frames_out[name] = rec

    report = {
        "frame_convention": "camera0_opencv (+X right, -Y up, +Z into scene); "
                            "each frame's object_pose (scalar-first quaternion + "
                            "translation + uniform scale) maps a canonical point "
                            "to that frame's camera. Compose the 4x4 T@R@S via "
                            "rig.lie.object_pose_matrix (no matrix is stored).",
        "canonical": "+Z up, centered at origin, longest dimension ~= 1",
        "scale": spec.scale,
        "reference_frame": spec.reference_frame,
        "parts": list(parts),
        "joint_defs": list(spec.joint_defs),
        "frames": frames_out,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[render_wrapper] wrote {out_path}")
