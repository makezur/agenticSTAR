"""serve.py — the resident render-worker loop (runs INSIDE Blender).

Entered from render_wrapper.main() when --serve is set, AFTER the one-time,
expensive setup (scene.clear -> build_from -> capture_canonical ->
apply_default_material -> make_camera -> configure_render). Instead of rendering
the --frames list once and exiting, this serves per-request renders on stdin
forever: build() geometry stays RESIDENT, and each request carries its own
pose / joints / camera / output dir. This is what amortizes the Blender startup +
build() cost the one-shot path (render.sh) pays on every invocation
(see sweep.md's "each process pays Blender startup + build()").

It is driven by pool/manager.py (the manager), which spawns one Blender per
worker and talks to it over pipes. This module owns ONLY the Blender side.

Transport (worker <-> manager), line-delimited over the worker's own pipes:
  * IN  (stdin):  one JSON request per line (see _handle below).
  * OUT (stdout): the render VIEWS print [render_wrapper] diagnostics to stdout,
                  so replies can't just be "the next stdout line". Every control
                  message is a single line prefixed with SENTINEL; the manager
                  scans for that prefix and treats every other line as a log.
Control messages:
  * {"event":"ready", "scene_sha1": ...}         once, after this loop starts
    (build() done -> the worker can accept requests).
  * {"event":"result", "id", "ok", "out"?, "seconds"?, "error"?}   per request.

Request schema (one JSON object per stdin line):
  {
    "id":     "<order id>",         # echoed back in the result
    "out":    "/abs/out/dir",       # MANAGER-allocated; views write here. The
                                    #   worker never allocates a pass dir (that
                                    #   race lives in core.pass_dirs.allocate)
                                    #   and never touches the shared mesh/pose.json.
    "frame":  "000040.jpg",         # a scene FRAMES key -> resolved EXACTLY as the
                                    #   one-shot path would (seeded/authored pose +
                                    #   Pi3X K). This is the byte-equivalent path.
    "views":  "match,depth",        # which VIEWS to run (default "match")
    "joints": {"door_hinge": -45}, # OPTIONAL: override this frame's joint states
                                    #   (MERGED onto them — a state is absolute, so a
                                    #   replace would straighten the unmentioned ones)
    "pose":   {...}                 # OPTIONAL: base-pose override (also serves a pose
                                    #   not in the scene; K falls back to the
                                    #   placeholder unless the frame is a scene frame).
                                    #   Honoured by EVERY view, including the
                                    #   MULTI-FRAME ones: for those it is applied to
                                    #   the named scene frame BEFORE the frame list
                                    #   resolves, so camera-seeded frames seed from
                                    #   the override too, and each frame's overridden
                                    #   placement is the P0 its canonical rotation is
                                    #   right-multiplied onto. This is the channel a
                                    #   `seed` uses for anything but sweep/apply.
    "sweep":  {...}                 # OPTIONAL: per-request sweep settings. Keys
                                    #   mirror the --sweep-* flags; worker-global
                                    #   argparse state is restored after the request.
                                    #   Scoring masks go HERE: "mask" and "hand_mask"
                                    #   are sweep keys, never top-level ones. the manager
                                    #   fills "hand_mask" from the run layout's
                                    #   hand_masks_dir when the request omits it;
                                    #   "hand_mask": "" opts a request out.
    "apply":  {"apply": "yaw:90"} # OPTIONAL: per-request `apply` view settings
                                    #   ("views":"apply"); the "apply" key mirrors the
                                    #   --apply flag. Pair with "sweep":{"mask":...}
                                    #   for the required scoring mask. Restored after.
    "oapply": {"oapply": "flips",   # OPTIONAL: per-request PER-FRAME `oapply` view
               "preset": "z:quarters",  # settings ("views":"oapply"): each frame picks
               "frames": "a,b",     #   its OWN winner from this candidate set.
               "masks_dir": "..."}  #   "oapply" mirrors --oapply (sets incl. 'flips'),
                                    #   "preset" mirrors --opreset (axis + angle range);
                                    #   both may be given (their UNION is scored).
                                    #   "frames" mirrors --frames and SCOPES the view
                                    #   to a SUBSET (the seam-flip check); masks_dir /
                                    #   hand_masks_dir mirror their flags (the manager
                                    #   fills them from the run layout when omitted).
                                    #   Restored after.
    "oapply_all": {"oapply": "flips",  # OPTIONAL: per-request `oapply_all` settings
                   "preset": "z:full"} #   ("views":"oapply_all") — the same keys, but
                                    #   ONE shared rotation for every frame: the verb
                                    #   for a build-orientation error. Use per-frame
                                    #   "oapply" when the frames disagree instead.
    "osweep": {"ranges": "...",     # OPTIONAL: per-request `osweep` view settings
               "frames": "a,b"}     #   ("views":"osweep") — a SEARCH per frame; same
                                    #   masks/frames/preset keys as "oapply" plus
                                    #   "ranges"/"angle_preset" mirroring
                                    #   --osweep-ranges/--osweep-angle-preset.
                                    #   Precedence: ranges > preset > angle_preset.
  }
Unknown TOP-LEVEL keys are rejected (ok:false), exactly like unknown keys inside
"sweep"/"apply", so a misplaced scoring key (e.g. a top-level "hand_mask") cannot
be dropped silently and downgrade the sweep gate from iou_visible to iou_raw.
A "shutdown" request ({"op":"shutdown"}) or EOF on stdin ends the loop cleanly.

The DOF grammars inside "ranges"/"apply"/"oapply" strings are defined by the
view docs, not here: camera-frame roll/yaw/pitch/dpx/dpy/tz + absolute
joint states in views/sweeps/sweep.md; object-canonical rx/ry/rz + the
flips panel in views/sweeps/osweep.md, views/sweeps/oapply.md (per frame) and
views/sweeps/oapply_all.md (one shared rotation).
"""

import json
import sys
import time

from core import filehash
from core import joints as joints_core
from core.sweep_family import SWEEP_VIEWS

# NOTE: rig.render/scene/lighting import bpy, so they are imported LAZILY inside
# the functions that run under Blender — this keeps `from pool.serve import
# SENTINEL` (used by the bpy-free pool/manager.py, and by tests) importable in
# the analysis env without pulling in Blender.

SENTINEL = "@@POOL@@ "


# The PANEL keys, in one place, merged into EVERY sweep-family view's map below.
# They are accepted at the top level too (where pool/manager.py reads them to decide
# the sheet); inside a view's settings object they reach the ENGINE, which is what
# decides how many candidates get rendered at all. `pool/panels.py._setting` reads
# both spellings for all of pool.panels.PANEL_VIEWS, so every one of those views must
# accept them here — a view whose map omits them makes the order fail ok:false
# instead of just changing its panel level. Merged rather than restated so the next
# view added cannot forget them.
_PANEL_ARG_MAP = {
    "visuals": "sweep_visuals",
    "waive_visual_budget": "waive_visual_budget",
}

_SWEEP_ARG_MAP = {
    **_PANEL_ARG_MAP,
    "mask": "sweep_mask",
    "hand_mask": "sweep_hand_mask",
    "hand_dilate": "sweep_hand_dilate",
    "quality": "sweep_quality",
    "ranges": "sweep_ranges",
    "space": "sweep_space",
    "refine": "sweep_refine",
    "refine_shrink": "sweep_refine_shrink",
    "topk": "sweep_topk",
    "depth_weight": "sweep_depth_weight",
    "timeout": "sweep_timeout",
    # WHERE THE GRID STARTS (a placement) and a directional shift applied to that
    # start before searching. Named for what they do: `pose_start` is the pose the
    # grid's offsets are measured from, `start_shift` moves it first.
    "pose_start": "sweep_pose_start",
    "start_shift": "sweep_start_shift",
    "candidates": "sweep_candidates",
    "grid_slice": "sweep_grid_slice",
    "dump_topk": "sweep_dump_topk",
    "joints": "sweep_joints",
    "angle_preset": "sweep_angle_preset",
    "strategy": "sweep_strategy",
    "de_max_evals": "sweep_de_max_evals",
    "de_popsize": "sweep_de_popsize",
    "de_seed": "sweep_de_seed",
    "de_mutation": "sweep_de_mutation",
    "de_recombination": "sweep_de_recombination",
    "polish": "sweep_polish",
    "polish_max_evals": "sweep_polish_max_evals",
}


def validate_sweep_range_shape(sweep, default_strategy="de"):
    """Refuse grid step counts when the effective strategy is DE.

    A three-value ``name:lo,hi,steps`` range is grid-shaped; silently discarding
    ``steps`` would turn it into a continuous DE search. Two-value DE bounds remain
    valid and DE remains the default; callers that supply a third value must opt
    into ``strategy:grid``.
    """
    if not isinstance(sweep, dict):
        raise ValueError("request 'sweep' must be a JSON object")
    explicit_strategy = sweep.get("strategy")
    if explicit_strategy is not None and str(explicit_strategy).lower() not in (
        "grid", "de"
    ):
        raise ValueError(
            "sweep.strategy must be 'grid' or 'de', got "
            f"{str(explicit_strategy).lower()!r}")
    strategy = str(explicit_strategy or default_strategy or "de").lower()
    if strategy != "de":
        return
    ranges = sweep.get("ranges")
    if not isinstance(ranges, str):
        return
    stepped = []
    for raw in ranges.split(";"):
        name, separator, values = raw.partition(":")
        if separator and len([value for value in values.split(",") if value != ""]) >= 3:
            stepped.append(name.strip() or raw)
    if stepped:
        raise ValueError(
            "grid step count supplied to DE for "
            + ", ".join(stepped)
            + "; set sweep.strategy to 'grid' or remove the third range value")


def _apply_sweep_overrides(a, req):
    """Apply whitelisted sweep settings and return values to restore afterward."""
    sweep = req.get("sweep")
    if sweep is None:
        return {}
    if not isinstance(sweep, dict):
        raise ValueError("request 'sweep' must be a JSON object")
    unknown = sorted(set(sweep) - set(_SWEEP_ARG_MAP))
    if unknown:
        raise ValueError("unknown sweep request key(s): " + ", ".join(unknown))
    validate_sweep_range_shape(
        sweep, default_strategy=getattr(a, "sweep_strategy", "de"))
    saved = {}
    for key, attr in _SWEEP_ARG_MAP.items():
        if key not in sweep:
            continue
        saved[attr] = getattr(a, attr)
        value = sweep[key]
        # the two keys an order may spell as a JSON OBJECT/LIST rather than a string:
        # the engine takes them as text (inline JSON or a path), so an order carrying
        # the readable dict form is dumped here. Missing a key from this tuple means
        # the engine receives a Python repr and fails to parse it.
        if key in ("pose_start", "candidates") and not isinstance(value, str):
            value = json.dumps(value)
        setattr(a, attr, value)
    return saved


# Every key a request may carry at the TOP level (schema in the module
# docstring; "op" is the shutdown escape hatch, "out" is manager-allocated).
# Anything else is rejected up front: a misplaced scoring key (e.g. "hand_mask"
# at the top level instead of inside "sweep") would otherwise be dropped
# SILENTLY, and the sweep would quietly gate on iou_raw with hand-occluded
# pixels scored as misses.
# "raw" / "visuals" / "waive_visual_budget" are PANEL keys: the manager reads
# them after the render to decide what sheets to build (pool/panels.py) and the
# worker ignores them. They are whitelisted here because an unknown top-level key
# is rejected, so an agent asking for raw renders must not get ok:false.
# The five sweep-family view names come from core.sweep_family so a new member
# cannot be accepted by the manager and rejected here (or the reverse).
# "seed" is a MANAGER key: pool.reseed.resolve_order_seed expands
# "<path>#<selector>" into sweep.pose_start + joints at claim time and replaces it with
# "seed_from" (provenance — which candidate this hop descended from, kept because
# the claimed order is deleted on recycle). Both are whitelisted so the one-shot
# render path accepts an order written for the pool, rather than an agent's hop
# working under the manager and failing ok:false off it.
# "seed_cross_frame" is the per-order AUTHORIZATION for a seed whose record belongs
# to another frame (pose carried verbatim). It rides beside "seed"/"seed_from" and
# survives on the claimed order so the hop stays self-describing.
_TOP_LEVEL_KEYS = frozenset((
    "id", "op", "out", "frame", "views", "pose", "joints", "view_index",
    "pi3x_view_index",   # old spelling of "view_index", still accepted
    "raw", "visuals", "waive_visual_budget", "seed", "seed_from",
    "seed_cross_frame", "provenance",
) + SWEEP_VIEWS)


def _check_provenance(value):
    if value is None:
        return
    if not isinstance(value, dict):
        raise ValueError("request 'provenance' must be a JSON object")
    allowed = {"sequence", "chunk", "iteration", "window", "submitted_by"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError("unknown provenance key(s): " + ", ".join(unknown))
    if "iteration" in value:
        iteration = value["iteration"]
        if type(iteration) is not int:
            raise ValueError("provenance.iteration must be an integer")
        if iteration < 0:
            raise ValueError("provenance.iteration must be non-negative")
    for key in allowed - {"iteration"}:
        if key in value and not isinstance(value[key], str):
            raise ValueError(f"provenance.{key} must be a string")


def _check_top_level_keys(req):
    unknown = sorted(set(req) - _TOP_LEVEL_KEYS)
    if unknown:
        hint = ""
        if any(k in _SWEEP_ARG_MAP for k in unknown):
            hint = (" (scoring keys like 'mask'/'hand_mask' belong INSIDE the "
                    "'sweep' object)")
        raise ValueError("unknown top-level request key(s): "
                         + ", ".join(unknown) + hint)
    _check_provenance(req.get("provenance"))


# per-request settings for the imperative `apply` view (mirrors _SWEEP_ARG_MAP).
# It reuses sweep's SCORING flags from the command line, but its panel level is its
# own per-order decision, so it carries the panel keys like every other view.
_APPLY_ARG_MAP = {**_PANEL_ARG_MAP, "apply": "apply"}

# per-request settings for the MULTI-FRAME object-centric views. "frames" mirrors
# --frames and is what scopes osweep/oapply/oapply_all to a SUBSET of the scene's frames
# (the window-seam flip check): the whole-run branch resolves ctx.frames through
# resolve_frames(a, ...), so overriding a.frames for the request IS the subset
# mechanism. masks_dir/hand_masks_dir mirror their flags — the one-shot path
# gets them on the command line, but the manager's worker command doesn't carry them,
# so a pooled order states them (or lets the manager fill them from the run layout).
# "preset" mirrors --opreset and is shared by BOTH maps (it is the one preset flag
# every object-centric view speaks); each view consumes it its own way — the search
# verb grids it, the imperative verb expands it to a labelled candidate set.
_OAPPLY_ARG_MAP = {
    **_PANEL_ARG_MAP,
    "oapply": "oapply",
    "preset": "opreset",
    "frames": "frames",
    "masks_dir": "masks_dir",
    "hand_masks_dir": "hand_masks_dir",
}
# oapply_all takes the SAME keys as per-frame oapply (it is the same candidate
# grammar, applied to every frame at once), so it shares the map. Its own object key
# keeps the two orders distinct in one request.
_OAPPLY_ALL_ARG_MAP = dict(_OAPPLY_ARG_MAP)

_OSWEEP_ARG_MAP = {
    **_PANEL_ARG_MAP,
    "ranges": "osweep_ranges",
    "angle_preset": "osweep_angle_preset",
    "preset": "opreset",
    "frames": "frames",
    "masks_dir": "masks_dir",
    "hand_masks_dir": "hand_masks_dir",
}


def _apply_map_overrides(a, req, req_key, arg_map):
    """Apply one whitelisted per-request settings object (req[req_key], keys per
    arg_map) onto the worker-global args; returns the values to restore after."""
    obj = req.get(req_key)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"request {req_key!r} must be a JSON object")
    unknown = sorted(set(obj) - set(arg_map))
    if unknown:
        raise ValueError(f"unknown {req_key} request key(s): "
                         + ", ".join(unknown))
    saved = {}
    for key, attr in arg_map.items():
        if key not in obj:
            continue
        saved[attr] = getattr(a, attr, "")
        setattr(a, attr, obj[key])
    return saved


def _apply_apply_overrides(a, req):
    """Apply whitelisted `apply`-view settings and return values to restore."""
    return _apply_map_overrides(a, req, "apply", _APPLY_ARG_MAP)


def _restore_overrides(a, saved):
    for attr, value in saved.items():
        setattr(a, attr, value)


def _emit(msg):
    """Write one control line (SENTINEL + compact JSON) to stdout and flush.

    Kept on its own line and flushed immediately so the manager, which reads the
    worker's stdout line by line, never blocks waiting on a buffered reply."""
    sys.stdout.write(SENTINEL + json.dumps(msg, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _framespec_for(req, a, spec, ref_name, resolve_frames, FrameSpec):
    """Resolve the FrameSpec for one request.

    Frame mode (the tested, byte-equivalent path): `frame` names a scene FRAMES
    key -> reproduce the one-shot FrameSpec via resolve_frames (same seeding and
    Pi3X K), then apply optional `pose`/`joints` overrides. A pose for a name
    outside the scene builds a FrameSpec directly; intrinsics then fall back to
    the placeholder because the request cannot restate K.

    `joints` MERGES onto the frame's states rather than REPLACING them. Joint
    states are ABSOLUTE and `transforms.forward_kinematics` defaults a joint that
    is missing from the dict to 0.0 (rest), so a wholesale replace silently
    STRAIGHTENED every joint the request did not mention: `{"door": 10}` on a frame
    that also declares `lid: 80` rendered the lid flat. A request names the joints
    it wants to change; the rest keep the frame's committed state.
    """
    frame = str(req.get("frame") or req.get("id") or "req")
    pose = req.get("pose")
    joints = req.get("joints")

    if frame in spec.frames:
        # frame mode — resolve exactly as render_wrapper's one-shot path does.
        saved = a.frames
        try:
            a.frames = frame
            fs = resolve_frames(a, spec, ref_name)[0]
        finally:
            a.frames = saved
        if pose is not None:
            fs.pose = pose
        if joints is not None:
            merged = dict(fs.joint_states or {})
            merged.update(joints)
            fs.joint_states = merged
        return fs

    # explicit-pose (or off-scene frame) mode — direct construct.
    if pose is None:
        raise ValueError(
            f"frame {frame!r} is not in the scene FRAMES and no explicit 'pose' "
            "was given; provide a scene frame name or a 'pose' dict.")
    # There is no committed frame to merge onto here, so `joints` is the WHOLE
    # articulation state and must name every articulated joint — the one path
    # where the scene-load gate cannot cover for the order. Refused rather than
    # defaulted for the usual reason: 0.0 is a move, not a hold.
    joints_core.require_complete_states(
        spec.joint_defs, joints or {},
        f"off-scene frame {frame!r} (explicit 'pose', no committed state to "
        "merge onto)")
    view_index = req.get("view_index", req.get("pi3x_view_index"))
    return FrameSpec(frame, pose, spec.scale, joints or {},
                     intrinsics=None, view_index=view_index,
                     camera_c2w=None, intr_provenance=None)


def _resolve_frames_with_overrides(req, a, spec, ref_name, resolve_frames,
                                   multi_frame):
    """The full FrameSpec list a whole-list view needs, honouring the request's
    top-level `pose`/`joints` override on the frame it names.

    The override lands on the SPEC ENTRY, BEFORE `resolve_frames` — load-bearing.
    `resolve_frames` seeds every frame with no authored pose FROM the reference pose via
    the known Pi3X camera motion, so overriding after resolution would move one frame
    and leave the other N-1 seeded from a pose it no longer has. Applied before, the
    seeding carries it through to every frame.

    `frame` names which entry it lands on (default: the reference). `joints` MERGES onto
    that entry's states, as in `_framespec_for`: a state is absolute, so a replace
    straightens every joint the request did not mention. The spec is restored after —
    the worker is resident and the next request must see the scene's own numbers.
    """
    pose = req.get("pose")
    joints = req.get("joints")
    if pose is None and joints is None:
        return resolve_frames(a, spec, ref_name)

    target = str(req.get("frame") or ref_name)
    if target not in spec.frames:
        raise ValueError(
            f"views {multi_frame} resolve the whole frame list from the scene, so a "
            f"'pose'/'joints' override must name a scene frame, and {target!r} is not "
            f"in FRAMES (known: {sorted(spec.frames)}). No off-scene mode for these.")
    entry = spec.frames[target]
    saved = dict(entry)
    try:
        if pose is not None:
            entry["pose"] = dict(pose)
        if joints is not None:
            merged = dict(entry.get("joints") or {})
            merged.update(joints)
            entry["joints"] = merged
        return resolve_frames(a, spec, ref_name)
    finally:
        spec.frames[target] = saved


def _missing_reports(ran, out_dir):
    """Which sweep-family views ran but left no `<stem>.json` — i.e. SKIPPED.

    The backstop for a whole failure CLASS, not one bug. Every guard in both sweep
    engines' `_prepare` prints its reason and returns before the first render (a
    missing scoring mask, no posed frame, no mesh parts, no sweepable joints, an
    empty candidate grid) — and a `return` reads to the caller exactly like a
    completed view, so the worker would report `ok: true` with an EMPTY render dir.

    A sweep-family view that did its job ALWAYS writes its report (that is what the
    report is), so "ran, wrote nothing" is a reliable proxy for "skipped" that costs
    one stat per view and needs no engine change. Only these views are checked: the
    per-frame views (match/depth/ids/…) write images under their own names and have
    no single required artifact.
    """
    import os as _os
    from core.sweep_family import SWEEP_VIEWS
    return [v for v in ran
            if v in SWEEP_VIEWS
            and not _os.path.isfile(_os.path.join(out_dir, f"{v}.json"))]


def _run_request(req, a, spec, objs, canonical, cam_obj, ref_name,
                 resolve_frames, FrameSpec, VIEWS):
    """Render one request into req['out']. Mirrors render_wrapper's per-frame loop
    body (pose -> bbox -> Context -> views -> restore) with a.out redirected to
    the manager-allocated output dir. Returns the result dict (never raises: a
    failing request must not take the resident worker down)."""
    import os
    from rig import render, scene
    from views import MULTI_FRAME_VIEWS, unknown_view_error
    rid = req.get("id")
    t0 = time.time()
    saved_out, saved_views = a.out, a.views
    saved_sweep = {}
    saved_apply = {}
    saved_oapply = {}
    saved_oapply_all = {}
    saved_osweep = {}
    posed = False
    try:
        _check_top_level_keys(req)
        # A `seed` normally never reaches here — the manager expands it at claim
        # time — but a whitelisted key that nothing acts on is a SILENT DROP, and
        # the drop would be the seed itself: the frame renders at its committed
        # pose, plausibly, and the hop is lost. So resolve it here too. Idempotent
        # (it deletes `seed` and leaves `seed_from`), so a manager-expanded order
        # passes straight through.
        from pool import reseed
        reseed.resolve_order_seed(req)
        out = req.get("out")
        if not out:
            raise ValueError("request has no 'out' directory")
        os.makedirs(out, exist_ok=True)
        a.out = out
        a.views = req.get("views") or "match"
        saved_sweep = _apply_sweep_overrides(a, req)
        saved_apply = _apply_apply_overrides(a, req)
        saved_oapply = _apply_map_overrides(a, req, "oapply", _OAPPLY_ARG_MAP)
        saved_oapply_all = _apply_map_overrides(a, req, "oapply_all",
                                                _OAPPLY_ALL_ARG_MAP)
        saved_osweep = _apply_map_overrides(a, req, "osweep", _OSWEEP_ARG_MAP)

        # osweep/oapply/oapply_all read ctx.frames, so they get the RESOLVED FrameSpec
        # list (each entry carrying its own Pi3X K), exactly like render_wrapper's
        # branch for them. Scoping to one frame is the order's `"frames"` key, which
        # resolve_frames honours via a.frames — not a hand-built one-element list,
        # which would lose the per-frame K that `place_match_camera` needs.
        multi_frame = [v for v in render.requested_views(a)
                       if v in MULTI_FRAME_VIEWS]
        frames = None
        if multi_frame:
            frames = _resolve_frames_with_overrides(req, a, spec, ref_name,
                                                    resolve_frames, multi_frame)
            fs = next((f for f in frames if f.name == ref_name), frames[0])
        else:
            fs = _framespec_for(req, a, spec, ref_name, resolve_frames,
                                FrameSpec)
        scene.pose_frame(objs, canonical, fs.pose, spec.scale, spec.joint_defs,
                         fs.joint_states,
                         where=f"order {rid!r}, frame {fs.name!r}")
        posed = True
        center, radius = _camera_bbox()
        ctx = render.Context(a, cam_obj, center, radius, frame=fs, frames=frames,
                             spec=spec, canonical=canonical)
        ran = []
        for name in render.requested_views(a):
            fn = VIEWS.get(name)
            if fn is None:
                raise ValueError(unknown_view_error(name, VIEWS))
            fn(ctx)
            ran.append(name)
        missing = _missing_reports(ran, out)
        if missing:
            raise RuntimeError(
                f"view(s) {', '.join(missing)} ran but wrote no report — they SKIPPED "
                "before the first render. The reason is printed above in this "
                "worker's log (a missing scoring mask, no posed frame, no sweepable "
                "joints, an empty candidate grid). Nothing was rendered, so this is "
                "a failed order, not an empty success.")
        result = {"event": "result", "id": rid, "ok": True, "out": out,
                  "views": ran, "seconds": round(time.time() - t0, 3)}
        if req.get("provenance") is not None:
            result["provenance"] = req["provenance"]
        return result
    except BaseException as e:  # noqa: BLE001 — report ANY failure, keep serving
        result = {"event": "result", "id": rid, "ok": False,
                  "error": f"{type(e).__name__}: {e}",
                  "seconds": round(time.time() - t0, 3)}
        if req.get("provenance") is not None:
            result["provenance"] = req["provenance"]
        return result
    finally:
        # restore per-request args and reset the parts to canonical so the next
        # request starts from a clean rest pose (same invariant as the one-shot
        # per-frame loop's trailing scene.restore_world).
        a.out, a.views = saved_out, saved_views
        # reverse application order: oapply, oapply_all and osweep may ALL override
        # a.frames (a co-requested set), so the later-applied map must be
        # unwound first for the earlier save to restore the true original.
        _restore_overrides(a, saved_osweep)
        _restore_overrides(a, saved_oapply_all)
        _restore_overrides(a, saved_oapply)
        _restore_overrides(a, saved_apply)
        _restore_overrides(a, saved_sweep)
        if posed:
            scene.restore_world(objs, canonical)


def _camera_bbox():
    """center, radius of the currently-posed object. match/depth ignore these,
    but a sweep/turntable view reads them, so populate the Context faithfully."""
    from rig import camera
    return camera.scene_bbox()


def serve_loop(a, spec, objs, canonical, cam_obj, ref_name,
               resolve_frames, FrameSpec):
    """Serve render requests on stdin until EOF / a shutdown request.

    resolve_frames + FrameSpec are passed in from render_wrapper (which is
    __main__ under Blender, so importing it back would re-exec it). VIEWS is
    imported here (lazily built, bpy-side)."""
    from rig import lighting
    from views import VIEWS

    # Lighting is added ONCE here — setup_lighting() creates a fresh world + 3 new
    # sun lamps every call and never removes the old ones (and ignores its
    # center/radius args), so calling it per request would leak datablocks and
    # over-light later renders. One call == a single-frame one-shot render's
    # lighting exactly (the suns are placed by direction from the origin, not from
    # the object bbox), which is what makes the byte-equivalence bar hold.
    lighting.setup_lighting(None, None)

    # Announce readiness (build() is done). scene_sha1 lets the manager detect a
    # scene.py edit and recycle the pool. render_wrapper stamped the same hash
    # into MANIFEST.json; we recompute it from a.scene the same way.
    _emit({"event": "ready", "scene_sha1": filehash.file_sha1_or_none(a.scene),
           "pid": _pid()})

    for line in _iter_stdin_lines():
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as e:
            _emit({"event": "result", "id": None, "ok": False,
                   "error": f"bad JSON request: {e}"})
            continue
        if not isinstance(req, dict):
            _emit({"event": "result", "id": None, "ok": False,
                   "error": "request must be a JSON object"})
            continue
        if req.get("op") == "shutdown":
            _emit({"event": "bye", "id": req.get("id")})
            break
        res = _run_request(req, a, spec, objs, canonical, cam_obj, ref_name,
                           resolve_frames, FrameSpec, VIEWS)
        _emit(res)


def _iter_stdin_lines():
    """Yield stdin lines one at a time (blocking), ending on EOF. Wrapped so the
    loop reads unbuffered enough that a request is picked up as soon as the
    manager writes it + '\\n'."""
    stream = sys.stdin
    while True:
        line = stream.readline()
        if line == "":   # EOF — manager closed our stdin
            return
        yield line


def _pid():
    import os
    return os.getpid()
