"""render_wrapper.py — thin dispatcher, runs INSIDE Blender (bpy) via render.sh.

Fixed harness rig. The agent never edits this file (nor anything under rig/ or
views/). It orchestrates the pipeline; the actual work lives in two packages:
  * rig/    — shared infrastructure (args, scene, camera, lighting, render,
              imaging, export, pose transforms). See rig/README.md.
  * views/  — one module + doc per render view (match, turntable, ids,
              crop, visibility, sweep, depth), dispatched by name.

Pipeline (main):
  1. clear the factory scene,
  2. exec the agent-authored scene.py + build() -> a SceneSpec (shared canonical
     parts, SCALE, JOINTS defs, and per-frame FRAMES),
  3. capture the canonical (rest) world matrices,
  4. resolve the frames to render (their known tracking cameras, if --tracking, and each
     frame's base object pose — an authored/measured per-frame pose if present,
     else seeded from the reference frame's camera motion),
  5. add a camera + neutral lighting + engine config,
  6. run the per-STATE turntable (coherence in each articulation state), then the
     per-FRAME views (match/depth/diagnostics) with the object posed for each,
  7. write per-frame poses + joint states to pose.json,
  8. optionally export the CANONICAL (rest-pose) GLB — the poses live in
     pose.json, NOT baked into the mesh — with optional voxel remesh.

The agent's scene.py is DECLARATIVE and MULTI-FRAME: build() creates the union of
named parts in a CANONICAL frame (+Z up, centered, longest dim ~= 1); SCALE is the
shared physical size; JOINTS are shared joint definitions; FRAMES gives each
observed frame's base pose + per-joint states (+ an optional 'moved' flag).
"""

import datetime
import json
import os
import shutil
import subprocess
import sys

# rig/, views/ and pool/ live next to this file; make them importable under Blender's
# bundled python (the script dir is not on sys.path when run via --python).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy  # noqa: E402

from core import captures, modules, pass_dirs, run_layout  # noqa: E402
from rig import args as rig_args, camera, export, lighting, render, scene  # noqa: E402
from rig import transforms  # noqa: E402
from views import MULTI_FRAME_VIEWS, VIEWS, unknown_view_error  # noqa: E402


class FrameSpec:
    """Per-frame render context: name, resolved pose (+shared scale), joint
    states, camera intrinsics, and (if known) the tracking view index + extrinsic."""

    def __init__(self, name, pose, scale, joint_states, intrinsics=None,
                 view_index=None, camera_c2w=None, intr_provenance=None):
        self.name = name
        self.pose = pose                  # {rotation_euler, translation, ...}
        self.scale = scale                # shared
        self.joint_states = joint_states  # {joint_name: value}
        self.intrinsics = intrinsics      # cam_math.NormIntrinsics or None
        self.view_index = view_index
        self.camera_c2w = camera_c2w
        # provenance of `intrinsics`: "explicit" | "tracking" | None (None ->
        # place_match_camera falls back to the iPhone placeholder and records
        # "iphone-placeholder"). Recorded in pose.json + sweep manifests.
        self.intr_provenance = intr_provenance

    @property
    def placement(self):
        """Full placement (pose + shared scale) for the pose diagnostics."""
        return transforms.frame_placement(self.pose, self.scale)


# --------------------------------------------------------------------------- #
# frame resolution
# --------------------------------------------------------------------------- #
def _requested_frames(a, spec):
    """Ordered frame names to render: --frames if given, else all of FRAMES."""
    if a.frames.strip():
        names = [n.strip() for n in a.frames.split(",") if n.strip()]
        for n in names:
            if n not in spec.frames:
                raise RuntimeError(f"--frames: {n!r} not in scene FRAMES "
                                   f"({', '.join(spec.frames)})")
        return names
    return spec.frame_names


def _tracking_view_map(tracking_dir):
    """basename-stem -> tracking view index, via the canonical capture reader."""
    keyframes = captures.load_keyframes(tracking_dir) or []
    out = {}
    for k in keyframes:
        stem = captures.stem_of(k.get("frame_name", ""))
        if stem:
            out[stem] = int(k["view"])
    return out


def resolve_frames(a, spec, ref_name):
    """Build a FrameSpec per requested frame.

    Base pose per frame:
      * the reference frame           -> its authored pose (the gauge);
      * any frame with an authored    -> that pose (a per-frame pose measured by
        'pose'                           measure_depth or hand-authored — it wins
                                         over the camera seed, regardless of the
                                         semantic 'moved' label);
      * otherwise                     -> SEEDED from the reference pose via the
                                         known tracking camera motion (no free params).
    Intrinsics per frame: tracking K[k] normalized, else None (placeholder).
    """
    names = _requested_frames(a, spec)
    ref_pose = spec.frames[ref_name]["pose"]

    cams = view_map = None
    if a.tracking:
        cams = camera.load_tracking_cameras(a.tracking)
        view_map = _tracking_view_map(a.tracking)

    n_frames = len(names)
    out = []
    for name in names:
        entry = spec.frames[name]
        view_index = intr = prov = c2w = None
        rel = None
        if cams is not None:
            stem = os.path.splitext(str(name))[0]
            if stem not in view_map:
                raise RuntimeError(
                    f"frame {name!r} not found in tracking keyframes.json")
            view_index = view_map[stem]
            ref_stem = os.path.splitext(str(ref_name))[0]
            ref_view = view_map[ref_stem]
            rel = camera.relative_extrinsic(cams["c2w"], ref_view, view_index)
            c2w = cams["c2w"][view_index]
            if cams["size"] is not None:
                intr = camera.norm_intrinsics(cams["intrinsics"][view_index],
                                                   cams["size"])
                prov = "tracking"

        # explicit --intrinsics/--camera-json K wins over the per-frame tracking
        # K (priority: explicit > tracking > iPhone placeholder). It applies to
        # ALL frames — a single measured K is a camera-0 property, not per-frame.
        if a.explicit_intr is not None:
            intr = a.explicit_intr
            prov = "explicit"
            if cams is not None and n_frames > 1 and name == ref_name:
                print("[render_wrapper] WARNING: explicit --intrinsics/--camera-json "
                      f"overrides the per-frame tracking K for all {n_frames} frames; "
                      "per-frame K differences are ignored.")

        # resolve the base pose:
        #   * the reference frame        -> its authored pose (the gauge);
        #   * ANY frame with a pose       -> that authored/seeded pose (a per-frame
        #                                    pose measured by measure_depth or hand-
        #                                    authored wins over the camera seed);
        #   * otherwise                   -> SEEDED from the reference via the known
        #                                    tracking camera motion (no free params).
        # `moved` is a SEMANTIC label (the base pose genuinely changed — read by
        # aggregate.py / the downstream URDF), NOT the switch that gates seeding:
        # an authored pose is honored regardless of `moved`.
        if name == ref_name:
            pose = ref_pose
        elif entry["pose"] is not None:
            pose = entry["pose"]
        elif entry["moved"]:
            # moved=True asserts the object was RE-POSED this frame, so the camera
            # seed (which assumes it did NOT move) is provably the wrong pose —
            # falling back to it would be a silent no-op indistinguishable from
            # moved=False. Require an explicit pose (measure_depth emits one per
            # frame; paste it, or hand-author).
            raise RuntimeError(
                f"frame {name!r}: moved=True requires an authored 'pose' (the "
                "object was re-posed, so it cannot use the camera seed). Author "
                "FRAMES[...]['pose'] (measure_depth's pose_snippet gives one per "
                "frame), or set moved=False to use the seeded pose verbatim.")
        else:
            # no per-frame pose -> pose is the camera seed used verbatim
            pose = (transforms.reseat_pose(ref_pose, rel, spec.scale)
                    if rel is not None else ref_pose)

        out.append(FrameSpec(name, pose, spec.scale, entry["joints"],
                             intrinsics=intr, view_index=view_index,
                             camera_c2w=c2w, intr_provenance=prov))
    return out


def _harness_rev():
    """The harness git revision, or None outside a checkout.

    Best-effort provenance, never fatal: `git` may be absent inside Blender's
    environment and a released harness may not be a checkout at all. `-dirty`
    matters more than the sha here — an uncommitted harness means the snapshot's
    imports cannot be pinned to anything.
    """
    try:
        out = subprocess.run(
            ["git", "-C", os.path.dirname(os.path.abspath(__file__)),
             "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _write_manifest(pass_dir, name, label, a, spec, ref_name, frame_names,
                    scene_record):
    """Record what this pass rendered, for provenance + freshness auditing.

    `scene_record` is snapshot_scene()'s return, taken back when the scene was
    read for the exec. It is threaded in rather than recomputed here BECAUSE
    here is too late: this runs after every render, so a fresh read would
    fingerprint whatever the file says now, not what produced these images.
    """
    renders_root = os.path.dirname(os.path.abspath(pass_dir.rstrip(os.sep)))
    manifest = {
        "pass": name,
        # The pass this one succeeds, so the chain is walkable from any link
        # WITHOUT trusting a number in a note. Paired with scene_snapshot it
        # turns scene_sha1 from an opaque tripwire into an index: the parent's
        # snapshot is the exact "before" to diff this pass against.
        "parent": pass_dirs.parent_of(renders_root, name),
        "label": label or None,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "views": render.requested_views(a),
        "frames": list(frame_names),
        "ref_frame": ref_name,
        "engine": a.engine,
        "samples": a.samples,
        "scene": os.path.abspath(a.scene),
        # Both from the one pre-exec read. Hashing the snapshot back MUST
        # reproduce scene_sha1 — a mismatch means the snapshot or the manifest
        # is corrupt and the lineage is not trustworthy.
        "scene_sha1": scene_record.get("scene_sha1"),
        "scene_snapshot": scene_record.get("scene_snapshot"),
        # The snapshot records the authored delta, which is what a diff needs,
        # but scene.py imports the harness (`from shapes import box`) so it is
        # NOT replayable on its own: re-execing it needs the harness at this
        # revision. Stamped so "revert to pass N's scene" can tell whether it
        # would even rebuild the same geometry.
        "harness_rev": _harness_rev(),
    }
    with open(os.path.join(pass_dir, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# --------------------------------------------------------------------------- #
# view classification
# --------------------------------------------------------------------------- #
# turntable is a per-STATE shape check (runs once over unique joint states, not
# per frame); every other view renders per frame with the object posed for it.
_TURNTABLE = "turntable"

# mechanism is a per-JOINT check (runs once over the declared joints, sweeping each
# across its limit). Dispatched in its own branch for the same reason as turntable:
# in the per-frame loop it would re-render the identical sweep once per frame — 30
# frames x 9 states of the same canonical arc — and the frame's pose would be
# discarded by the view anyway (it renders in the CANONICAL frame, deliberately;
# see views/mechanism.md).
_MECHANISM = "mechanism"

# osweep/oapply/oapply_all are called ONCE with the resolved frames list (which --frames
# may scope to a single frame) and write ONE report, so they are dispatched in their own
# branch like turntable, never in the per-frame loop. Membership is shared with
# pool/serve.py via views.MULTI_FRAME_VIEWS.
_MULTI_FRAME = MULTI_FRAME_VIEWS


def _resolve_explicit_intrinsics(a):
    """Parse --intrinsics / --camera-json ONCE into a resolution-agnostic
    NormIntrinsics on a.explicit_intr (+ a.explicit_intr_wh for the default
    canvas). Rejects a non-identity camera-json 'pose' (the harness uses a fixed
    camera-0 extrinsic and moves the object, not the camera). None when neither
    flag is given."""
    a.explicit_intr = None
    a.explicit_intr_wh = None
    intr = None
    if a.camera_json:
        with open(a.camera_json) as f:
            src = json.load(f)
        pose = src.get("pose")
        if pose is not None:
            mat = camera.parse_pose(pose)
            if any(abs(mat[i][j] - (1.0 if i == j else 0.0)) > 1e-6
                   for i in range(4) for j in range(4)):
                raise RuntimeError(
                    f"--camera-json {a.camera_json}: a non-identity 'pose' is not "
                    "supported. The harness renders from a FIXED camera-0 (identity "
                    "extrinsic) and moves the OBJECT per frame — supply intrinsics "
                    "only (drop 'pose'), and express viewpoint via the object pose.")
        intr = src.get("intrinsics")
        if isinstance(intr, str):
            intr = camera.parse_intrinsics_str(intr)
    if a.intrinsics:
        intr = camera.parse_intrinsics_str(a.intrinsics)
    if intr is None:
        return
    a.explicit_intr = camera.cam_math.NormIntrinsics.from_pixels(
        intr["fx"], intr["fy"], intr["cx"], intr["cy"],
        intr["width"], intr["height"], provenance="explicit")
    a.explicit_intr_wh = (int(intr["width"]), int(intr["height"]))
    print(f"[render_wrapper] explicit intrinsics fx={intr['fx']} fy={intr['fy']} "
          f"cx={intr['cx']} cy={intr['cy']} @ {int(intr['width'])}x"
          f"{int(intr['height'])} (overrides the tracking K for all frames)")


def main():
    a = rig_args.parse_args()
    _resolve_explicit_intrinsics(a)

    # Each invocation gets its own pass bucket under the renders root (a.out);
    # every view roots its output at a.out, so redirecting it once here sends the
    # whole pass into the bucket. The pass dir is printed at the end — that exact
    # path is what the agent reads this iteration.
    renders_root = a.out

    # Safety net for a manual render.sh that forgot --tracking: if no real K was given
    # (neither --tracking nor --intrinsics/--camera-json), auto-resolve the run's
    # tracking dir from RUN_DIR/layout.json's "capture" entry (a walk-up from the
    # renders root, the same shape as the sweep's depth_config.json resolution).
    # Without this, a sweep silently scores every candidate under the iPhone-13
    # PLACEHOLDER K. Explicit flags still win; --no-auto-tracking forces the
    # placeholder. Provenance stays "tracking", and we log the source, so it is
    # never silent.
    if not a.tracking and a.explicit_intr is None and not getattr(a, "no_auto_tracking", False):
        auto_tracking = run_layout.resolve_tracking(renders_root)
        if auto_tracking:
            a.tracking = auto_tracking
            print(f"[render_wrapper] auto-resolved --tracking from layout.json "
                  f"capture: {auto_tracking} (pass --no-auto-tracking to force the "
                  f"placeholder K)")

    pass_label = pass_dirs.normalize_label(a.pass_label)
    pass_name, pass_dir = pass_dirs.allocate(renders_root)
    a.out = pass_dir

    # Snapshot + fingerprint the scene HERE — before the exec below, from one
    # read — so both describe the bytes that actually produced this pass's
    # renders. Deferring either to _write_manifest (which runs after every
    # render) would record whatever scene.py says minutes later instead.
    scene_record = pass_dirs.snapshot_scene(
        pass_dir, a.scene,
        warn=lambda msg: print(f"[render_wrapper] {msg}"))

    # Same timing, same reason: written post-render it would be lost for
    # exactly the interrupted pass that needs it.
    pass_dirs.write_intent(
        pass_dir, a.intent, a.rollback,
        warn=lambda msg: print(f"[render_wrapper] {msg}"))

    scene.clear()
    spec = scene.build_from(a.scene)

    objs = scene.mesh_parts()
    canonical = scene.capture_canonical(objs)

    ref_name = a.ref_frame.strip() or spec.reference_frame
    if ref_name not in spec.frames:
        raise RuntimeError(f"--ref-frame {ref_name!r} not in FRAMES "
                           f"({', '.join(spec.frames)})")

    lighting.apply_default_material()
    cam_obj = camera.make_camera(a.fov, a.ortho)
    render.configure_render(a)

    # Resident worker mode (the render pool): the one-time setup above
    # (clear -> build_from -> capture_canonical -> material/camera/engine) is done;
    # instead of rendering the frame list and exiting, serve per-request renders on
    # stdin forever. build() geometry stays resident; each request carries its own
    # pose/joints/camera. resolve_frames + FrameSpec are passed in so pool/serve.py
    # need not import this module (which is __main__ here).
    if getattr(a, "serve", False):
        from pool.serve import serve_loop
        serve_loop(a, spec, objs, canonical, cam_obj, ref_name,
                   resolve_frames, FrameSpec)
        return

    frames = resolve_frames(a, spec, ref_name)
    requested = render.requested_views(a)
    # A run without the mechanism module never renders the A/B arcs — not even
    # on request: the sheets would exist with no gate to answer, and the agent
    # would read them as owed. Refuse loudly so a saved order learns the state.
    if _MECHANISM in requested and not modules.enabled_at(a.out, "mechanism"):
        sys.exit("[render_wrapper] " + modules.disabled_line("mechanism")
                 + " — drop 'mechanism' from --views")
    per_frame_views = [v for v in requested
                       if v not in (_TURNTABLE, _MECHANISM)
                       and v not in _MULTI_FRAME]

    # which frame the single-frame diagnostics (sweep/ids/visibility/crop)
    # run against: --frame, else the reference frame.
    diag_frame = a.frame.strip() or ref_name

    # --- per-STATE turntable (shape coherence in each articulation state) ----
    if _TURNTABLE in requested:
        # pose at the reference frame so lighting/framing have a sensible scene,
        # then the turntable view re-poses per unique state internally. A --frames
        # subset may exclude the reference frame — fall back to the first resolved
        # frame rather than dying with a bare StopIteration before any output.
        ref_fs = next((f for f in frames if f.name == ref_name), frames[0])
        scene.pose_frame(objs, canonical, ref_fs.pose, spec.scale,
                         spec.joint_defs, ref_fs.joint_states,
                         where=f"turntable framing pose, frame {ref_fs.name!r}")
        center, radius = camera.scene_bbox()
        lighting.setup_lighting(center, radius)
        ctx = render.Context(a, cam_obj, center, radius, frame=ref_fs,
                             spec=spec, canonical=canonical)
        VIEWS[_TURNTABLE](ctx)
        scene.restore_world(objs, canonical)

    # --- per-JOINT mechanism sweep (does each joint move as declared?) --------
    # Lighting is set from the reference frame's framing, as above, so the sweep is
    # lit like every other view; the view itself re-poses in the CANONICAL frame
    # (no base pose) and frames its own camera to the whole swept extent.
    if _MECHANISM in requested:
        ref_fs = next((f for f in frames if f.name == ref_name), frames[0])
        scene.pose_frame(objs, canonical, ref_fs.pose, spec.scale,
                         spec.joint_defs, ref_fs.joint_states,
                         where=f"mechanism framing pose, frame {ref_fs.name!r}")
        center, radius = camera.scene_bbox()
        lighting.setup_lighting(center, radius)
        ctx = render.Context(a, cam_obj, center, radius, frame=ref_fs,
                             spec=spec, canonical=canonical)
        VIEWS[_MECHANISM](ctx)
        scene.restore_world(objs, canonical)

    # --- MULTI-FRAME views (osweep/oapply per frame; oapply_all shared) -------
    # These score canonical rotations right-multiplied onto every frame's pose, so
    # they need the FULL resolved frames list (not just one). Pose at the reference
    # for a sensible framing/lighting scene; the view re-poses each frame internally
    # and restores the scene afterward (non-destructive).
    multi_frame = [v for v in requested if v in _MULTI_FRAME]
    if multi_frame:
        ref_fs = next((f for f in frames if f.name == ref_name), frames[0])
        scene.pose_frame(objs, canonical, ref_fs.pose, spec.scale,
                         spec.joint_defs, ref_fs.joint_states,
                         where=f"multi-frame framing pose, frame {ref_fs.name!r}")
        center, radius = camera.scene_bbox()
        lighting.setup_lighting(center, radius)
        ctx = render.Context(a, cam_obj, center, radius, frame=ref_fs,
                             frames=frames, spec=spec, canonical=canonical)
        for name in multi_frame:
            VIEWS[name](ctx)
        scene.restore_world(objs, canonical)

    # --- per-FRAME views (match/depth/diagnostics) ---------------------------
    for fs in frames:
        # diagnostics only run for the chosen diag frame (they are single-frame
        # tools); match/depth run for every frame.
        views_here = [v for v in per_frame_views
                      if v in ("match", "depth") or fs.name == diag_frame]
        if not views_here:
            continue
        scene.pose_frame(objs, canonical, fs.pose, spec.scale, spec.joint_defs,
                         fs.joint_states, where=f"frame {fs.name!r}")
        center, radius = camera.scene_bbox()
        lighting.setup_lighting(center, radius)
        ctx = render.Context(a, cam_obj, center, radius, frame=fs, spec=spec,
                             canonical=canonical)
        for name in views_here:
            fn = VIEWS.get(name)
            if fn is None:
                print("[render_wrapper] WARNING: "
                      + unknown_view_error(name, VIEWS))
                continue
            fn(ctx)
        scene.restore_world(objs, canonical)

    # --- serialize per-frame poses + joint states ----------------------------
    part_names = [o.name for o in objs]
    base_w, base_h = a.base_w, a.base_h
    frame_results = {
        fs.name: {
            "pose": fs.pose,  # the RESOLVED pose (seeded frames aren't in the spec)
            "matrix": transforms.compose_placement_with_scale(fs.pose, spec.scale),
            "view_index": fs.view_index,
            "camera_c2w": fs.camera_c2w,
            # intrinsics provenance + the effective pixel K at the base render
            # resolution (a NormIntrinsics is resolution-agnostic; None -> the
            # iPhone-13 placeholder was used).
            "intr_provenance": fs.intr_provenance or "iphone-placeholder",
            "intrinsics": (fs.intrinsics.at(base_w, base_h)
                           if fs.intrinsics is not None
                           else camera.iphone13_intrinsics(base_w, base_h)),
        }
        for fs in frames
    }
    pose_out = os.path.join(pass_dir, "pose.json")
    scene.write_pose_json(pose_out, spec, frame_results, part_names)

    if a.export:
        latest_pose = os.path.join(os.path.dirname(a.export), "pose.json")
        os.makedirs(os.path.dirname(latest_pose), exist_ok=True)
        shutil.copy2(pose_out, latest_pose)
        print(f"[render_wrapper] copied latest pose to {latest_pose}")
        # parts are already at canonical (restored after each frame); export the
        # rest-pose mesh — poses live in pose.json, not baked into vertices.
        scene.restore_world(objs, canonical)
        export.export_glb(a.export, a.remesh)

    # record what this pass rendered.
    _write_manifest(pass_dir, pass_name, pass_label, a, spec, ref_name,
                    [fs.name for fs in frames], scene_record)
    # Announce the exact pass dir loudly — this is THE path the agent reads/writes
    # this iteration (its renders + where analysis.viz.composite/analysis.scorers.silhouette should point).
    print("[render_wrapper] " + "=" * 60)
    print(f"[render_wrapper] PASS {pass_name} OUTPUT DIR: {pass_dir}")
    print(f"[render_wrapper]   rendered {','.join(requested) or 'none'} "
          f"for frames {','.join(fs.name for fs in frames)}")
    print(f"[render_wrapper]   read this pass's renders from: {pass_dir}/")
    print("[render_wrapper] " + "=" * 60)
    print("[render_wrapper] done")


if __name__ == "__main__":
    main()
