"""runtime.py — the Blender-side RUNTIME GLUE for the `sweep` view.

The sweep machinery is split into logical blocks (see this package's __init__):
the pure candidate/space/metrics logic lives in `dof_space`/`planner`/`metrics`;
the render loop lives in `engine`. This module is what is left once those moved
out — the small Blender-facing pieces the engine leans on:

  * `Progress`      — throttled stderr progress + a pre-flight ETA that AUTO-CAPS a
                      runaway grid, and the sweep's deadline owner (expired()).
  * `make_deadline` — the wall-clock deadline Progress owns.
  * `DepthSupervision` — observed-depth supervision: silhouette IoU is depth-blind (a
                      nearer+smaller pose projects the same outline), so with
                      --tracking each scoring render also writes its Z pass (same render
                      call) and the candidate's RAW conf-weighted depth error joins
                      the ranking. Reported in CANONICAL units (fraction of the
                      object's longest dimension), so 0.23 reads as "off by a
                      quarter of the object" for any object.
  * `base_scale` / `resize_nearest` / `format_duration` — small helpers
    the engine + depth supervision share.

The camera-frame "order" geometry (the `OrderCtx`) now lives with the DOF space in
`dof_space.py`; the combined-score `score_of` and landscape stats live in `metrics.py`.
"""

import glob
import os
import sys
import time

import bpy
import numpy as np

from core import depth_config
from rig import imaging, lie, raster, render, scene


def make_deadline(timeout):
    """Wall-clock deadline `now + timeout` seconds, or None when timeout<=0.

    The sweep views compute this once in render_view and hand it to `Progress`,
    which owns the deadline (both the caller's --*-timeout and any auto-cap) so the
    scoring loop breaks between candidates via `prog.expired()` (granularity is
    one candidate — a single in-flight render can't be cut).
    """
    t = float(timeout or 0.0)
    return (time.time() + t) if t > 0 else None


def format_duration(seconds):
    """Human 'Xs' / 'Ym Zs' / 'Hh Mm' for a projected wall-clock (pre-flight ETA)."""
    s = max(0.0, float(seconds))
    if s < 90:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{int(s // 60)}m {int(s % 60):02d}s"
    return f"{int(s // 3600)}h {int((s % 3600) // 60):02d}m"


# Projected wall-clock (from the first candidate) above which a sweep that was
# given NO explicit --*-timeout auto-caps itself to this budget: it announces the
# cap and still ranks + writes the best-so-far (flagged timed_out), so a big grid
# can never again silently run for tens of minutes with nothing to show. An
# explicit --*-timeout always takes precedence (the agent asked for the cost up
# front — this is the safety net when they didn't set one).
ETA_AUTOCAP_SECONDS = 600.0


class Progress:
    """Throttled stderr progress + pre-flight ETA + deadline owner for the sweeps.

    One instance per view run; update() is called once per scored candidate
    across ALL _score_candidates passes (coarse + refine), so a big grid stays
    visibly alive without a line per candidate. `total` is an estimate
    (grid_size * (1 + refine_passes)) — a guide, not a bound: the printed
    fraction may read past `total` if a refine grid differs, and done() prints
    the true count.

    On the FIRST candidate it emits a PRE-FLIGHT line: the total candidate count
    times the just-measured per-candidate wall-clock = a projected total, printed
    BEFORE the grid grinds on (the reported "silent for 10 minutes" bug). If that
    projection blows past `autocap_seconds` and the run was given no explicit
    timeout (`user_timeout <= 0`), it self-imposes a deadline at that budget so
    best-so-far is still written. Progress owns `deadline` so the loops just call
    `expired()` and never re-derive the timeout logic three times.
    """

    # Project after this many candidates. The FIRST candidate pays one-time warmup
    # (first EEVEE render compiles shaders + builds the BVH) and sweep's opener can
    # be a near-free behind-camera skip, so both are excluded: the estimate is the
    # mean of candidates 2..N, which tracks steady-state cost. Still well under ~1s.
    _PREFLIGHT_AFTER = 4

    def __init__(self, total, label, deadline=None, user_timeout=0.0,
                 autocap_seconds=ETA_AUTOCAP_SECONDS, interval=1.0,
                 stream=sys.stderr):
        self.total = max(1, int(total))
        self.label = label
        self.interval = float(interval)
        self.stream = stream
        self.n = 0
        self.best = 0.0
        self.start = time.time()
        self._last = 0.0
        self.deadline = deadline                     # abs wall-clock or None
        self._user_timeout = float(user_timeout or 0.0)
        self._autocap = float(autocap_seconds or 0.0)
        self._preflighted = False
        self._warm_t = None                          # time the 1st candidate finished

    def expired(self):
        """True once the (user or auto-cap) deadline has passed — the loops'
        between-candidate break condition. None deadline -> never expires."""
        return self.deadline is not None and time.time() >= self.deadline

    def _preflight(self):
        """Project total wall-clock from the steady-state per-candidate time and,
        if over budget with no explicit timeout, auto-cap the deadline. Excludes
        the warmup candidate (self._warm_t): the mean over candidates 2..n."""
        self._preflighted = True
        base_t = self._warm_t if self._warm_t is not None else self.start
        n_measured = max(1, self.n - (1 if self._warm_t is not None else 0))
        per = (time.time() - base_t) / n_measured      # steady-state s/candidate
        projected = per * self.total
        line = (f"[render_wrapper] {self.label}: PRE-FLIGHT ~{per:.2f}s/candidate "
                f"x {self.total} candidates (grid+refine) = "
                f"~{format_duration(projected)} projected")
        if (self._autocap > 0 and self._user_timeout <= 0
                and projected > self._autocap):
            self.deadline = self.start + self._autocap
            line += (f". OVER the {format_duration(self._autocap)} budget and no "
                     f"--*-timeout set -> AUTO-CAPPING at {format_duration(self._autocap)} "
                     "(best-so-far still written, flagged timed_out). Set a timeout, "
                     "fewer steps/DOFs, or higher --*-quality to control this")
        self.stream.write(line + "\n")
        self.stream.flush()

    def update(self, best_gate, force=False):
        self.n += 1
        if best_gate is not None and best_gate > self.best:
            self.best = best_gate
        if self.n == 1:
            self._warm_t = time.time()   # exclude the warmup candidate from the ETA
        if not self._preflighted and self.n >= min(self._PREFLIGHT_AFTER, self.total):
            self._preflight()
        now = time.time()
        if force or (now - self._last) >= self.interval:
            self._last = now
            self.stream.write(
                f"[render_wrapper] {self.label}: {self.n}/{self.total} scored, "
                f"best {self.best:.4f}, {now - self.start:.1f}s\n")
            self.stream.flush()

    def done(self):
        self.stream.write(
            f"[render_wrapper] {self.label}: done {self.n}/{self.total} scored, "
            f"best {self.best:.4f}, {time.time() - self.start:.1f}s\n")
        self.stream.flush()


# --------------------------------------------------------------------------- #
# observed-depth supervision — per-candidate raw depth error on the scoring grid
# --------------------------------------------------------------------------- #
# confidence below the per-run floor gets weight 0 (untrustworthy observed
# pixels don't penalise), mirroring analysis.scorers.depth. The resolver lives in
# bpy-free core/ (core.depth_config), shared with the analysis env — re-exported
# here for the engines.
_resolve_conf_thr = depth_config.resolve_conf_thr
depth_cost_enabled = depth_config.depth_cost_enabled

# dtype-agnostic nearest-neighbour resample — one implementation, in imaging.
resize_nearest = imaging.resize_nearest


def resolve_depth_weight(cfg_weight, out_dir):
    """The effective depth-supervision weight (clamped >= 0).

    `cfg_weight` None means AUTO: resolve the per-run depth_config.json's
    `weight` (the backend's default), else 0.1 — so a noisy estimator backend
    can supervise more gently than exact GT depth with no per-order flag. An
    explicit value (--sweep-depth-weight / an order's depth_weight) wins.
    Forced to 0 when the "cost" flip switch is off — the sweep then ranks on
    pure IoU with no DepthSupervision built at all."""
    if not depth_cost_enabled(out_dir):
        return 0.0
    return max(0.0, depth_config.resolve_depth_weight(out_dir,
                                                      override=cfg_weight))


class ScoringSession:
    """The non-destructive low-res scoring lifecycle BOTH sweep engines share.

    A sweep must leave the scene bit-identical for the views that follow it in
    the same pass (match/turntable/pose.json/GLB). This object owns that
    contract once: snapshot world matrices + camera/render state on entry;
    switch to the low-res scoring camera + cheap scoring settings + optional
    GPU rasterizer for the candidate loop; then tear the scoring rig down and
    restore everything. The engines keep only their own math (which candidates,
    which frames, how to rank).

    Usage:
        sess = ScoringSession(ctx.cam_obj, objs, a, quality, a.out, STEM, view=V)
        sess.begin_scoring()              # low-res + samples=1 + raster
        ... pose_frame(...); sess.score_posed(grid_iou, depth_sup) ...
        sess.end_scoring()                # free raster, drop scratch, restore
        sess.set_full_res()               # winner re-render at match res
        ... render_to(...) ...
        sess.restore()                    # world + camera back (idempotent)
    """

    def __init__(self, cam_obj, objs, args, quality, out_dir, stem, view):
        self.cam_obj = cam_obj
        self.objs = objs
        self.args = args
        self.view = view
        self.quality = quality
        self.scratch = os.path.join(out_dir, f"_{stem}_scratch.png")
        self.rast = None
        self._saved_world = {o: o.matrix_world.copy() for o in objs}
        self._saved_scoring = None
        self._saved_render = render.snapshot_camera_render(cam_obj)
        self.sw, self.sh = render.sweep_resolution(args.base_w, args.base_h,
                                                   quality)

    # ---- scoring rig ------------------------------------------------------ #
    def begin_scoring(self):
        """Low-res resolution + cheap scoring settings + optional GPU raster."""
        bpy.context.scene.render.resolution_x = self.sw
        bpy.context.scene.render.resolution_y = self.sh
        self._saved_scoring = render.snapshot_scoring_state()
        render.apply_scoring_state(samples=1)
        self.rast = raster.maybe_create(self.cam_obj, self.objs, self.sw,
                                        self.sh, label=self.view)
        if self.rast is not None:
            print(f"[render_wrapper] {self.view}: scoring via GPU raster "
                  f"(SWEEP_GPU_RASTER=0 for per-candidate EEVEE renders)")
        return self.rast

    def score_posed(self, grid_iou, depth_sup=None):
        """Score the CURRENTLY POSED scene: (iou_raw, iou_visible, depth_mae,
        depth_canon). One rast-vs-EEVEE branch for both engines — the raster
        reads camera-Z off the framebuffer; the EEVEE path renders a silhouette
        PNG (+ the DepthSupervision compositor EXR rider when depth is on)."""
        depth_mae = depth_canon = None
        if self.rast is not None:
            if depth_sup is not None:
                rmask, zgrid = self.rast.silhouette_and_depth()
                depth_mae, depth_canon = depth_sup.score_grid(rmask, zgrid)
            else:
                rmask = self.rast.silhouette()
        else:
            rmask = render.render_silhouette(self.scratch)
            if depth_sup is not None:
                depth_mae, depth_canon = depth_sup.score(rmask)
        iou_raw, iou_visible = grid_iou.score(rmask)
        return iou_raw, iou_visible, depth_mae, depth_canon

    def end_scoring(self):
        """Free the raster, drop the scratch render, restore scoring settings."""
        if self.rast is not None:
            self.rast.free()
            self.rast = None
        if os.path.isfile(self.scratch):
            os.remove(self.scratch)
        if self._saved_scoring is not None:
            render.restore_scoring_state(self._saved_scoring)
            self._saved_scoring = None

    # ---- scene state ------------------------------------------------------ #
    def set_full_res(self):
        """Back to the full match resolution (the winner re-render)."""
        bpy.context.scene.render.resolution_x = self.args.base_w
        bpy.context.scene.render.resolution_y = self.args.base_h

    def restore(self):
        """World matrices + camera/render state back to entry (idempotent —
        both engines restore mid-way for the winner re-render and again on
        exit)."""
        scene.restore_world(self.objs, self._saved_world)
        render.restore_camera_render(self.cam_obj, self._saved_render)


def base_scale(scale):
    """The object's longest canonical dimension in scene units: the shared,
    uniform scalar SCALE. Canonical longest dim ~= 1, so dividing a depth error
    by this reads as 'fraction of the object's size' — a natural metric
    prior (0.23 of a microwave = clearly wrong; 0.06 = fine) that holds for any
    object. Always the BASE scale, never a candidate's swept scale, so a
    candidate can't shrink its relative error by inflating its own scale."""
    return max(lie.scalar_scale(scale, "shared SCALE"), 1e-9)


class DepthSupervision:
    """Per-candidate RAW depth error vs the OBSERVED pointmap, riding the
    scoring render.

    The Z pass costs no extra render: setup() enables it plus a compositor File
    Output node writing a float32 EXR into a scratch dir during the SAME
    bpy.ops.render.render call that produces the silhouette PNG; score() reads it
    back per candidate. The error is the conf-weighted mean |render_Z - observed_Z|
    over (object mask ∧ rendered geometry ∧ observed keep ∧ conf >= thr ∧ ¬hand) —
    deliberately RAW, no scale fit: a fitted scale assumes the pose is right,
    so a pose error would masquerade as 'scale'. Raw
    error is the honest pose-sensitive number, reported in canonical units
    (fraction of the object's longest dimension, see base_scale).

    Non-destructive: teardown() removes the nodes it added and restores the
    compositor/Z-pass flags (same pattern as views/depth.py), so match /
    turntable / depth outputs are unaffected.
    """

    def __init__(self, obs_depth, weights, scale, tmp_dir):
        self.obs_depth = obs_depth   # (H,W) float32 at the scoring grid
        self.weights = weights       # (H,W) float32 conf, 0 where not scorable
        self.scale = scale           # base_scale() of the scene's shared SCALE
        self.tmp_dir = tmp_dir
        self._saved = None
        self._nodes = ()

    @classmethod
    def maybe_create(cls, tracking_dir, view_index, mask_bool, keep, grid_hw,
                     scale, out_dir, label):
        """A DepthSupervision for this sweep, or None (no --tracking / no observed
        depth for the frame / unreadable arrays -> the sweep ranks by IoU
        alone). `tracking_dir` is the capture's tracking dir; the per-frame depth
        npz lives beside it in <capture>/depth/<stem>.npz (numpy-only reader
        from core.captures — works in Blender's python)."""
        if not tracking_dir or view_index is None:
            return None
        from core import captures
        v = int(view_index)
        try:
            stem = None
            for k in captures.load_keyframes(tracking_dir) or []:
                if int(k["view"]) == v:
                    stem = captures.stem_of(k.get("frame_name", ""))
                    break
            if stem is None:
                raise FileNotFoundError(f"no keyframes entry for view {v}")
            depth_path = os.path.join(
                os.path.dirname(tracking_dir.rstrip(os.sep)),
                captures.DEPTH_DIR, stem + ".npz")
            _, depth, conf, pkeep, _ = captures.load_depth(depth_path)
        except Exception as e:
            print(f"[render_wrapper] {label}: depth supervision unavailable "
                  f"({e}); ranking by IoU alone")
            return None
        # everything onto the SCORING grid, once per sweep (not per candidate).
        depth = resize_nearest(depth, grid_hw)
        conf = resize_nearest(conf, grid_hw)
        pkeep = resize_nearest(pkeep, grid_hw)
        m = resize_nearest(mask_bool, grid_hw)
        conf_thr = _resolve_conf_thr(out_dir)
        scorable = m & pkeep & (conf >= conf_thr)
        if keep is not None:  # waive the (dilated) hand region, like iou_visible
            scorable &= resize_nearest(keep, grid_hw)
        weights = np.where(scorable, conf, 0.0).astype(np.float32)
        if float(weights.sum()) <= 0.0:
            print(f"[render_wrapper] {label}: depth supervision has no "
                  "scorable pixels (mask/conf/hand leave nothing); ranking by "
                  "IoU alone")
            return None
        return cls(depth, weights, scale,
                   os.path.join(out_dir, f"_{label}_depth_tmp"))

    def setup(self):
        """Enable the Z pass + a File Output EXR rider on the scoring renders.

        Unlike views/depth.py (which never reads the still), the sweep KEEPS
        reading the scratch PNG for the silhouette — with the compositor on but
        no Composite node, that still would come out black. So a passthrough
        Composite node (Image + Alpha) keeps the PNG identical to a plain render
        while the File Output node writes the Z pass beside it."""
        scn = bpy.context.scene
        vl = bpy.context.view_layer
        self._saved = {"use_nodes": scn.use_nodes, "use_pass_z": vl.use_pass_z}
        vl.use_pass_z = True
        scn.use_nodes = True
        tree = scn.node_tree
        rl = tree.nodes.new("CompositorNodeRLayers")
        comp = tree.nodes.new("CompositorNodeComposite")
        comp.use_alpha = True
        tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
        tree.links.new(rl.outputs["Alpha"], comp.inputs["Alpha"])
        fout = tree.nodes.new("CompositorNodeOutputFile")
        fout.format.file_format = "OPEN_EXR"
        fout.format.color_mode = "RGB"
        fout.format.color_depth = "32"
        os.makedirs(self.tmp_dir, exist_ok=True)
        fout.base_path = self.tmp_dir
        fout.file_slots[0].path = "depth"
        tree.links.new(rl.outputs["Depth"], fout.inputs[0])
        self._nodes = (rl, comp, fout)
        scn.frame_set(0)  # File Output appends the frame number -> depth0000.exr
        # background pixels carry clip_end as their Z; anything in the far half
        # of the frustum is no real geometry (mirrors views/depth.py).
        self._far = 0.5 * float(scn.camera.data.clip_end)

    def score(self, rmask_bool):
        """(depth_mae, depth_mae_canon) for the candidate just rendered.

        `rmask_bool` is the candidate's silhouette at the scoring resolution
        (render_silhouette's alpha>=0.5) — it gates coverage, so no second EXR
        slot is needed for alpha. (None, None) when nothing scorable overlaps
        (score_of then applies the full penalty)."""
        hits = glob.glob(os.path.join(self.tmp_dir, "depth*.exr"))
        if not hits:
            return None, None
        img = bpy.data.images.load(hits[0], check_existing=False)
        w, h = int(img.size[0]), int(img.size[1])
        ch = img.channels
        px = imaging.image_pixels(img).reshape(h, w, ch)
        bpy.data.images.remove(img)
        depth = px[::-1, :, 0]  # bottom-up -> top-down, R channel = Z
        return self._score_arrays(rmask_bool, depth, self._far)

    def score_grid(self, rmask_bool, depth):
        """score() for an IN-MEMORY camera-Z grid (the GPU-raster path): same
        conf-weighted MAE, no compositor rider / EXR round trip. `depth` is a
        top-down float32 (H, W) with background at clip_end — exactly what the
        Z pass carried, so the same far-half cutoff separates real geometry."""
        far = 0.5 * float(bpy.context.scene.camera.data.clip_end)
        return self._score_arrays(rmask_bool, depth, far)

    def _score_arrays(self, rmask_bool, depth, far):
        if depth.shape != self.obs_depth.shape:
            depth = resize_nearest(depth, self.obs_depth.shape)
            rmask_bool = resize_nearest(rmask_bool, self.obs_depth.shape)
        valid = rmask_bool & (depth < far) & (self.weights > 0)
        w_v = self.weights[valid]
        tot = float(w_v.sum())
        if tot <= 0.0:
            return None, None
        err = np.abs(depth[valid] - self.obs_depth[valid])
        mae = float((err * w_v).sum() / tot)
        return mae, mae / self.scale

    def teardown(self):
        """Remove the rider nodes + restore flags (call before the winner
        re-render, so sweep_best.png doesn't write a stray full-res EXR)."""
        scn = bpy.context.scene
        if scn.node_tree is not None:
            for node in self._nodes:
                try:
                    scn.node_tree.nodes.remove(node)
                except Exception:
                    pass
        self._nodes = ()
        if self._saved is not None:
            scn.use_nodes = self._saved["use_nodes"]
            bpy.context.view_layer.use_pass_z = self._saved["use_pass_z"]
            self._saved = None
        for leftover in glob.glob(os.path.join(self.tmp_dir, "*")):
            os.remove(leftover)
        if os.path.isdir(self.tmp_dir):
            os.rmdir(self.tmp_dir)
