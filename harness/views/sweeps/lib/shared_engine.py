"""shared_engine.py — the OBJECT-CENTRIC reorientation engine (osweep/oapply/oapply_all).

The sweep/apply engine (`engine.py`) scores CAMERA-frame pose increments
LEFT-multiplied onto ONE frame's pose, against ONE mask — it fixes a single frame
whose pose drifted. This engine is the other axis: it scores rotations in the
object's own CANONICAL frame, RIGHT-multiplied onto each frame's pose — in
PER_FRAME mode about that frame's own FK-AABB centre
(M_k' = M_k @ T(c_k)·R_extra·T(-c_k), the same material point engine.py orbits;
lib/pivot.py), in SHARED mode about the canonical origin (M_k' = M_k @ R_extra;
see _prepare for why that is deliberate) — rendering + scoring the whole frame
list in one call. It re-ORIENTS the object; it never re-places it (the pivoted
form moves `t` only by the pivot correction that keeps the object in place on
screen) and never moves the camera.

TWO MODES, and which verb is which matters — `config.py`'s `mode` comment is the
authoritative statement:

  * PER_FRAME (`osweep`, `oapply`) — one rotation PER FRAME, each frame taking its
    OWN argmax by its own gate IoU. Semantically N independent sweep/apply runs that
    happen to share a Blender process and a candidate set: nothing is aggregated,
    nothing inherited. Reach for these when the frames DISAGREE — a window seam, one
    frame stuck in a flipped basin.
  * SHARED (`oapply_all`) — ONE rotation for every frame, ranked by MEAN gate IoU.
    That is the fix for an object BUILT mis-oriented, so it reads wrong in every
    frame: one decision, one shared canonical rotation, re-orienting the geometry
    coherently across all the differing camera views at once.

  `osweep` SEARCHES a grid of canonical rotations (the rx/ry/rz rotation vector,
  agent-facing DEGREES); `oapply` is its imperative sibling, applying named
  rotations and reporting the same per-frame IoU table.

Maximal reuse: the shared rotation math is `rig.lie.canonical_rotation` +
`right_reorient`; scoring reuses `rig.imaging` (mask/IoU), `rig.render`
(scoring-res + snapshot/restore), `rig.scene.pose_frame`, the sweep's `metrics`
(score_of, paste/format helpers, depth verdict) and `runtime` (Progress,
DepthSupervision). Only the multi-frame loop, the 3-DOF canonical grid, and the
multi-frame report are new here.

Non-destructive, like every sweep view: world matrices + camera/render/scoring state
are snapshotted and restored, so match/turntable/pose.json/GLB are unaffected. The
durable channel is the per-authored-frame paste block.
"""

import os
import shutil

import bpy
import numpy as np

from rig import imaging, lie, render, scene, transforms
from rig.transforms import _mat_to_np
from core import visual_budget
from views.sweeps.lib import dof_space, metrics, pivot, planner
from views.sweeps.lib.planner import CANON_DOFS
# the config + the mode constants are pure and live in config; re-exported here
# because every caller reaches them as shared_engine.SharedConfig / .PER_FRAME.
from views.sweeps.lib.config import (PER_FRAME, SHARED,    # noqa: F401
                                     SharedConfig)         # noqa: F401
from views.sweeps.lib.shared_placement import (canon_dict as _canon_dict,
                                               canon_quat as _canon_quat,
                                               placement as _placement)
# the two report writers: bpy-free JSON/text formatting.
from views.sweeps.lib.shared_report import (
    write_report as _write_report,
    write_per_frame_report as _write_per_frame_report)
from views.sweeps.lib.runtime import (
    base_scale as _base_scale,
    DepthSupervision as _DepthSupervision,
    make_deadline as _deadline,
    Progress as _Progress,
    resolve_depth_weight as _resolve_depth_weight,
    ScoringSession as _ScoringSession)


# A named candidate SET (oapply flip panel / custom '|' list) re-renders every
# candidate per frame at full match res when the set is at most this big — a flip
# panel is 4 candidates, and per-candidate renders are what the visual flip
# judgment needs. Bigger sets fall back to winner-only (grid behavior).
#
# The number itself lives in `core/visual_budget.py` with the other image-count
# policy, and the docs that quote it point at it there by name.
_MAX_SET_RENDERS = visual_budget.MAX_SET_RENDERS


def _best_for_frame(raw, prep, depth_weight):
    """One frame's own argmax over the candidates IT scored, or None.

    `raw` is the {cand: {frame: rec}} matrix score_pass builds. This is the whole of
    per-frame selection: the frame's own score_of, its own gate field, its own depth
    supervision — the same call `engine` makes for a single-frame sweep, which is
    what "per-frame osweep is N sweeps" means concretely."""
    mine = [(c, recs[prep.fs.name]) for c, recs in raw.items()
            if prep.fs.name in recs]
    if not mine:
        return None
    return max(mine,
               key=lambda cr: metrics.score_of(cr[1], prep.gate_field,
                                                depth_weight))[0]


# --------------------------------------------------------------------------- #
# per-frame preparation
# --------------------------------------------------------------------------- #
class _FramePrep:
    """Everything scoring one frame needs, resolved once before the candidate loop:
    the base lie.Pose, its mask grid + gate field, its rotation pivot, and (with
    --tracking) its depth supervision arrays keyed by the frame's own view_index."""

    def __init__(self, fs, P0, mask_bool, keep, gate_field, authored):
        self.fs = fs
        self.P0 = P0                # lie.Pose of the frame's resolved base pose
        self.mask_bool = mask_bool
        self.keep = keep
        self.gate_field = gate_field
        self.authored = authored
        # PER_FRAME mode: the canonical-frame FK-AABB pivot the rotation turns
        # about, measured at THIS frame's articulation (set in _prepare; None in
        # SHARED mode, where the rotation stays about the canonical origin —
        # see _prepare for why). `pivot` is its wire record.
        self.c_canon = None
        self.pivot = None


def _frame_base_pose(fs, scale):
    """The frame's resolved base pose as a lie.Pose (uniform scale recovered)."""
    M = _mat_to_np(transforms.compose_placement_with_scale(fs.pose, scale))
    return lie.pose_from_matrix(M)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
# `run` is the SETUP both modes share, then a two-way branch. The two modes are
# genuinely different searches — SHARED ranks candidates by their MEAN across the
# frames and refines one basin; PER-FRAME is N independent sweeps with N winners and
# N refine windows — so they are two functions, `_run_shared` and `_run_per_frame`,
# not one function with `if per_frame` at every step. What they share is everything
# up to the first candidate render, which `_prepare` resolves once into a `_Run`.
class _Run:
    """Everything both modes need, resolved before the first render.

    A plain attribute bag rather than a tuple: the setup produces ~15 values, and a
    positional handoff between three functions is where a refactor like this goes
    wrong. Nothing here is mode-specific — that is the point of the split."""

    def __init__(self, ctx, cfg, sess, objs, preps, candidates, labels, search,
                 ranges, canonical, scale, joint_defs, ref_name, depth_weight,
                 refine_n, prog):
        self.ctx, self.cfg, self.sess = ctx, cfg, sess
        self.a = ctx.args
        self.V, self.STEM = cfg.view, cfg.out_stem
        self.objs, self.preps = objs, preps
        self.candidates, self.labels = candidates, labels
        # `search` is what the grid comes from; `ranges` is its LATTICE view (see
        # _prepare). They differ only for a preset's discrete turn set.
        self.search, self.ranges = search, ranges
        self.canonical, self.scale = canonical, scale
        self.joint_defs, self.ref_name = joint_defs, ref_name
        self.depth_weight, self.refine_n, self.prog = depth_weight, refine_n, prog
        self.sw, self.sh = sess.sw, sess.sh
        # the candidate list every frame gets in a shared pass / the coarse pass
        self.same = {p.fs.name: candidates for p in preps}
        self.timed_out = False
        # the search PASSES, recorded as they run. Shared mode refines ONE basin, so
        # it keeps a single list; per-frame mode refines a window per frame, so it
        # keeps one list per frame — see `record_pass`. Either way the report can say
        # which lattice a candidate came from, which is exactly what a coarse-only
        # `ranges` field could not: a refined winner sits off that grid by design.
        self.passes = []
        self.frame_passes = {}

    # ---- the render/score primitives, shared verbatim by both modes ---------- #
    def score_pass(self, cands_by_frame, rast):
        """Render one pass of (frame, candidate) pairs -> {cand: {frame: rec}}.

        Keyed by FRAME on the way in because per-frame mode refines each frame into
        its own window, so pass 2 asks different candidates of different frames;
        shared mode just hands every frame the same list."""
        a, ctx = self.a, self.ctx
        by_cand = {}
        bpy.context.scene.render.resolution_x = self.sw
        bpy.context.scene.render.resolution_y = self.sh
        for prep in self.preps:
            if self.prog.expired():
                break
            cands = list(cands_by_frame.get(prep.fs.name) or ())
            if not cands:
                continue
            render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius,
                                      prep.fs.intrinsics)
            if rast is not None:
                rast.update_camera()   # this frame's K -> a new projection
            depth_sup = None
            if self.depth_weight > 0.0:
                depth_sup = _DepthSupervision.maybe_create(
                    a.tracking, getattr(prep.fs, "view_index", None), prep.mask_bool,
                    prep.keep, (self.sh, self.sw), _base_scale(self.scale), a.out,
                    f"{self.STEM}:{os.path.splitext(str(prep.fs.name))[0]}")
            if depth_sup is not None and rast is None:
                depth_sup.setup()   # compositor EXR rider (EEVEE path only)
            try:
                for c in cands:
                    if self.prog.expired():
                        break
                    pose = self.pose_for(prep, c)
                    rec = {"pose": pose}
                    (rec["iou_raw"], rec["iou_visible"],
                     rec["depth_mae"], rec["depth_canon"]) = \
                        self.sess.score_posed(prep.grid_iou, depth_sup)
                    by_cand.setdefault(c, {})[prep.fs.name] = rec
                    self.prog.update(self.prog.best)
            finally:
                if depth_sup is not None and rast is None:
                    depth_sup.teardown()
        return by_cand

    def pose_for(self, prep, cand):
        """Pose the rig for one (frame, candidate) and return the pose dict.

        The ONE place the canonical rotation meets the scene: both the scoring pass
        and the full-res panel loops go through it, so a change to how a candidate
        becomes geometry cannot apply to only one of them. `prep.c_canon` is the
        frame's FK-AABB pivot in PER_FRAME mode and None (canonical origin) in
        SHARED mode — set once in _prepare, which owns the why."""
        pose = _placement(prep.P0, _canon_quat(cand), centre=prep.c_canon)
        # `where=`: pose_frame refuses an incomplete joint-state dict, and here the
        # states are the FRAME's (these verbs never search articulation), so the frame
        # is the only useful thing the message can name — with N frames scored in one
        # run, "pose_frame" alone would not say which one is under-declared.
        scene.pose_frame(self.objs, self.canonical, pose, self.scale,
                         self.joint_defs, prep.fs.joint_states,
                         where=f"{self.V}: frame {prep.fs.name!r}, candidate {cand}")
        return pose

    def score_of(self, rec, prep):
        """This frame's combined score for one rec (its own gate field + depth)."""
        return metrics.score_of(rec, prep.gate_field, self.depth_weight)

    def refine_window(self, window, best_c):
        """One coarse->fine step: shrink `window` around `best_c` -> a new grid."""
        window = planner.refine_canon_ranges(window, best_c,
                                              float(self.cfg.refine_shrink))
        return window, planner.canon_grid(window)

    def record_coarse(self, n, frame=None):
        """Record pass 0 — the coarse lattice, a named candidate SET (which has
        none), or an so3 draw (no lattice either, so the record carries the
        GENERATOR — {n, seed} — instead: the one way the grid stays re-drawable
        from the report alone). `frame` scopes it to one frame's pass list in
        per-frame mode."""
        so3 = None
        kind = "candidates" if self.labels else "coarse"
        if isinstance(self.search, planner.So3Plan):
            kind, so3 = "so3", {"n": self.search.n, "seed": self.search.seed}
        self._passes_for(frame).append(metrics.pass_record(
            0, self.ranges, n, kind=kind, so3=so3))

    def record_refine(self, window, n, best_c, frame=None):
        """Record one refine pass: the lattice it scored, how many it got through, and
        the candidate its window re-centered on. `window` is the SHRUNK band, not
        `self.ranges` — recording the coarse band here is the bug this closes."""
        self._passes_for(frame).append(metrics.pass_record(
            len(self._passes_for(frame)),
            planner.canon_report_ranges(window), n, kind="refine",
            center=_canon_dict(best_c), shrink=float(self.cfg.refine_shrink)))

    def _passes_for(self, frame):
        if frame is None:
            return self.passes
        return self.frame_passes.setdefault(frame, [])

    def begin_panels(self):
        """Leave scoring res, return to full match res, and resolve the panel knobs
        both modes hand `metrics.panel_plan`."""
        self.sess.restore()
        self.sess.set_full_res()
        canon_kinds = {d: dof_space.POSE for d in CANON_DOFS
                       if d in (self.ranges or {})}
        return (visual_budget.visuals_level(self.cfg.visuals),
                bool(self.cfg.waive_visual_budget), canon_kinds)

    def render_panel(self, prep, cand, rank_of, winner):
        """Render one panel (frame x candidate) and return its filename."""
        self.pose_for(prep, cand)
        img = panel_image_name(self.STEM, cand,
                               os.path.splitext(str(prep.fs.name))[0],
                               self.labels, rank_of, winner)
        render.render_to(os.path.join(self.a.out, img))
        return img

    def best_image(self, frame_name, img):
        """The winner always answers to <stem>_best_<frame>.png, labelled or not: the
        report points at that name. A labelled panel keeps its label name too, so the
        winner is COPIED rather than renamed."""
        if not self.labels:
            return img
        best_img = f"{self.STEM}_best_{os.path.splitext(str(frame_name))[0]}.png"
        shutil.copy2(os.path.join(self.a.out, img),
                     os.path.join(self.a.out, best_img))
        return best_img

    def report_args(self):
        """The frame bookkeeping both writers take, identically.

        Subset detection keys off the REQUESTED frames (--frames), not the scored
        preps: a frame dropped for a missing mask must not silently flip the whole
        report into subset mode (its skip is already printed loudly)."""
        ctx = self.ctx
        return dict(
            depth_weight=self.depth_weight, sweep_res=[self.sw, self.sh],
            match_res=[self.a.base_w, self.a.base_h], timed_out=self.timed_out,
            labels=self.labels, scene_path=self.a.scene,
            requested_frames=[fs.name for fs in ctx.frames],
            scene_frames=(list(ctx.spec.frames) if ctx.spec is not None
                          else [fs.name for fs in ctx.frames]))

    def nothing_scored(self):
        """Every candidate failed or the clock ran out: restore and say so."""
        self.sess.restore()
        print(f"[render_wrapper] {self.V}: no candidates scored "
              f"({'timed out' if self.timed_out else 'empty'}); nothing written")


def _resolve_preps(ctx, cfg, ref_name, scale):
    """The frames that can be scored, one `_FramePrep` each; frames with no mask are
    skipped LOUDLY (a silent drop reads as a frame that scored badly).

    Mask paths join by frame stem through the ONE owner of that convention
    (core.captures.mask_for — numpy-only, so it is safe in Blender's python)."""
    from core import captures
    preps = []
    for fs in ctx.frames:
        mask_path = captures.mask_for(cfg.masks_dir, str(fs.name))
        if not os.path.isfile(mask_path):
            print(f"[render_wrapper] {cfg.view}: no mask for {fs.name!r} at "
                  f"{mask_path}; skipping this frame")
            continue
        hand_path = ""
        if cfg.hand_masks_dir:
            cand = captures.mask_for(cfg.hand_masks_dir, str(fs.name))
            hand_path = cand if os.path.isfile(cand) else ""
        mask_bool, keep = imaging.load_mask_grid(mask_path, hand_path,
                                                 cfg.hand_dilate)
        gate_field = "iou_visible" if keep is not None else "iou_raw"
        # (grid_iou is attached per prep AFTER sw/sh are known, in _prepare)
        authored = (fs.name == ref_name or
                    (ctx.spec is not None
                     and ctx.spec.frames.get(fs.name, {}).get("pose") is not None))
        preps.append(_FramePrep(fs, _frame_base_pose(fs, scale), mask_bool, keep,
                                gate_field, authored))
    return preps


def _resolve_candidates(cfg):
    """-> (candidates, labels, search, ranges).

    An explicit candidate SET (oapply's flip panel / custom '|' list) bypasses the
    grid; labels ride along for filenames + the report. Grid mode keeps labels None
    (osweep's continuous search has no meaningful names).

    `search` is what the grid comes from (a plain range dict, or a planner CanonPlan
    when --opreset carried sample LISTS); `ranges` is its LATTICE view — what the
    report records and what panel binning uses. They differ only for a preset, where
    a discrete turn set has samples but no refinable band."""
    if cfg.candidates:
        return ([tuple(c) for _, c in cfg.candidates],
                {tuple(c): str(lab) for lab, c in cfg.candidates}, {}, {})
    # PRECEDENCE (--osweep-ranges > --opreset > --osweep-angle-preset) lives in
    # planner.resolve_canon_search so it is pure and testable.
    search = planner.resolve_canon_search(cfg.ranges, cfg.preset, cfg.angle_preset,
                                           render.clamp_quality(cfg.quality))
    return planner.canon_grid(search), None, search, \
        planner.canon_report_ranges(search)


def _prepare(ctx, cfg):
    """Resolve everything both modes share -> (`_Run`, rast), or (None, None) if the
    run cannot proceed (each refusal prints its own reason and restores state).

    Everything here is mode-agnostic by construction: masks, base poses, the
    candidate grid, the scoring camera, depth supervision, the progress budget."""
    a = ctx.args
    V, STEM = cfg.view, cfg.out_stem
    if not cfg.masks_dir or not os.path.isdir(cfg.masks_dir):
        print(f"[render_wrapper] {V}: --masks-dir <dir of per-frame masks> is "
              f"required (got {cfg.masks_dir!r}); skipping {V}")
        return None, None
    if not ctx.frames:
        print(f"[render_wrapper] {V}: no frames resolved; skipping")
        return None, None
    objs = scene.mesh_parts()
    if not objs:
        print(f"[render_wrapper] {V}: no mesh parts to pose; skipping")
        return None, None

    scale = ctx.spec.scale
    joint_defs = ctx.spec.joint_defs if ctx.spec is not None else []
    ref_name = a.ref_frame.strip() or (ctx.spec.reference_frame
                                       if ctx.spec is not None else None)
    canonical = ctx.canonical or scene.capture_canonical(objs)
    quality = render.clamp_quality(cfg.quality)
    sess = _ScoringSession(ctx.cam_obj, objs, a, quality, a.out, STEM, view=V)

    preps = _resolve_preps(ctx, cfg, ref_name, scale)
    if not preps:
        print(f"[render_wrapper] {V}: no frames have a mask under {cfg.masks_dir}; "
              f"skipping {V}")
        sess.restore()
        return None, None

    # the ONE mode resolution in this setup (also used by the announce line below).
    per_frame = str(cfg.mode) == PER_FRAME
    # PER_FRAME rotations turn about each frame's own FK-AABB pivot (lib/pivot.py
    # — the same material point the camera-frame grammar orbits), measured at THAT
    # frame's articulation and frozen across candidates/refine passes so a pasted
    # winner reproduces its scored render bit-for-bit. SHARED mode (oapply_all)
    # deliberately keeps the canonical ORIGIN: its contract is ONE canonical-space
    # correction — equivalent to rebuilding the object rotated — and a shared
    # right-factor is what lets seeded frames inherit the reference's paste; a
    # per-frame pivot there would silently turn one decision into N different ones.
    if per_frame:
        pivot_cache = {}   # frames sharing an articulation share the measurement
        for prep in preps:
            key = tuple(sorted((str(k), float(v))
                               for k, v in (prep.fs.joint_states or {}).items()))
            if key not in pivot_cache:
                pivot_cache[key] = pivot.frame_pivot(
                    joint_defs, prep.fs.joint_states, canonical, objs)
            prep.c_canon, prep.pivot = pivot_cache[key]

    candidates, labels, search, ranges = _resolve_candidates(cfg)
    if not candidates:
        print(f"[render_wrapper] {V}: empty candidate grid; skipping")
        sess.restore()
        return None, None

    # ---- low-res scoring camera + non-destructive state --------------------- #
    sw, sh = sess.sw, sess.sh
    rast = sess.begin_scoring()
    # per-frame precomputed exact IoU (identical numbers to iou_on_grid)
    for prep in preps:
        prep.grid_iou = imaging.GridIoU(prep.mask_bool, prep.keep, (sh, sw))

    # depth supervision honors the per-run depth_config.json "cost" flip switch.
    depth_weight = _resolve_depth_weight(cfg.depth_weight, a.out)

    deadline = _deadline(cfg.timeout)
    # a named candidate SET is discrete — there is no basin to refine into.
    refine_n = 0 if labels else max(0, int(cfg.refine))
    # per-frame refine is candidates x frames x passes with NO shared basin to
    # exploit (each frame descends alone), which is exactly the cost of running N
    # sweeps — the same number the shared estimate already comes to, since shared
    # mode also re-scores every frame each pass.
    prog = _Progress(len(candidates) * len(preps) * (1 + refine_n), V,
                     deadline=deadline, user_timeout=cfg.timeout)

    if labels:
        what = f"candidate set [{', '.join(labels[c] for c in candidates)}]"
    elif isinstance(search, planner.So3Plan):
        # name the generator, not the DOF list: "over rx,ry,rz" would read like a
        # default box when this is a Haar draw with no lattice at all.
        what = (("per-frame" if per_frame else "shared")
                + f" re-localise: identity + {search.n} Haar-uniform SO(3) "
                + f"rotation(s) (seed {search.seed}, baked)")
    else:
        what = (("per-frame" if per_frame else "shared")
                + " canonical rotation over "
                + ",".join(planner.canon_searched_dofs(search)))
    print(f"[render_wrapper] {V}: {what} at {sw}x{sh}, "
          f"{len(preps)} frame(s), {len(candidates)} candidate(s)"
          + (", ONE ROTATION PER FRAME" if per_frame else "")
          + (f", depth-aware (w={depth_weight:g})" if depth_weight else "") + ".")

    return _Run(ctx, cfg, sess, objs, preps, candidates, labels, search, ranges,
                canonical, scale, joint_defs, ref_name, depth_weight, refine_n,
                prog), rast


def require_rotations(a, view):
    """True if the imperative canonical verbs have something to apply, else print the
    named skip and return False.

    `oapply` and `oapply_all` are IMPERATIVE — they render the rotations you name and
    nothing else — so with neither flag set there is no work, and the skip has to say
    which flags would have given it some. Shared because both verbs ask the identical
    question and the message spells out the whole grammar; two copies is two places
    for the grammar to drift out of step with `planner_canon`."""
    if a.oapply.strip() or a.opreset.strip():
        return True
    print(f"[render_wrapper] {view}: --oapply 'dof:value;...' | 'flips' | "
          f"'flip:x|y|z' (or --opreset 'z:quarters') is required; skipping {view}")
    return False


def run(ctx, cfg):
    """Score a canonical object rotation across ALL frames; write <stem>.json /
    <stem>.txt / <stem>_best_<frame>.png. One entry point for both modes: SHARED
    (`oapply_all` — ONE rotation for the whole run, ranked by MEAN gate IoU) and
    PER_FRAME (`osweep`/`oapply` — N independent fits, each frame its own argmax)."""
    r, rast = _prepare(ctx, cfg)
    if r is None:
        return
    if str(cfg.mode) == PER_FRAME:
        _run_per_frame(r, rast)
    else:
        _run_shared(r, rast)


# --------------------------------------------------------------------------- #
# SHARED mode: ONE rotation, ranked by its MEAN across the frames
# --------------------------------------------------------------------------- #
def _run_shared(r, rast):
    """Search for the single rotation that scores best ACROSS all frames.

    The mean is the ranking key, which is what makes this mode meaningful and also
    what makes it a different search from `_run_per_frame`: one basin to refine into,
    one winner, one paste block per authored frame with the rest inheriting."""
    V = r.V

    def aggregate(by_cand):
        """Per candidate: mean + min gate-combined across the frames it scored."""
        out = {}
        for c, frames in by_cand.items():
            combos = [r.score_of(frames[p.fs.name], p)
                      for p in r.preps if p.fs.name in frames]
            if not combos:
                continue
            out[c] = {"mean": float(np.mean(combos)), "min": float(np.min(combos)),
                      "frames": frames}
            r.prog.update(out[c]["mean"])
        return out

    # ---- coarse pass + optional coarse->fine refinement into ONE basin ------- #
    scored = aggregate(r.score_pass(r.same, rast))
    r.record_coarse(len(scored))
    window = r.search
    for p in range(r.refine_n):
        if r.prog.expired() or not scored:
            break
        best_c = max(scored, key=lambda c: scored[c]["mean"])
        window, cands = r.refine_window(window, best_c)
        print(f"[render_wrapper] {V}: refine pass {p + 1}/{r.cfg.refine} "
              f"{len(cands)} candidates around best mean "
              f"{scored[best_c]['mean']:.4f}")
        before = len(scored)
        scored.update(aggregate(r.score_pass({p.fs.name: cands
                                              for p in r.preps}, rast)))
        # `len(scored) - before` and not `len(cands)`: a refine grid can re-score a
        # candidate the coarse pass already holds (the window nests inside it), and a
        # timeout can cut the pass short. This counts what the pass actually ADDED.
        r.record_refine(window, len(scored) - before, best_c)

    r.prog.done()
    r.sess.end_scoring()
    r.timed_out = r.prog.expired()
    if not scored:
        r.nothing_scored()
        return
    ranked = sorted(scored.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    best_c, best = ranked[0]

    # ---- restore + re-render at full match res ------------------------------ #
    # ONE panel policy, shared with the sweep family: metrics.panel_plan decides.
    # A named candidate SET of <= _MAX_SET_RENDERS renders EVERY candidate per frame
    # (<stem>_<label>_<frame>.png) -- flip twins routinely tie on IoU, and the whole
    # point of a flip panel is that the agent LOOKS instead of trusting the score.
    # GRID mode bins like any other sweep (best per ordered rx/ry/rz cell) instead of
    # collapsing to the winner alone.
    visuals, waived, canon_kinds = r.begin_panels()
    # binning records: the canon DOFs live in the candidate TUPLE, so wrap each in
    # the {order: {...}} shape value_of reads, carrying the tuple back out.
    bin_recs = [{"order": _canon_dict(c), "_cand": c} for c, _ in ranked]

    def plan(level):
        return metrics.panel_plan(
            bin_recs, canon_kinds, r.ranges, r.cfg.dump_topk, visuals=level,
            waived=waived, what=V, set_limit=_MAX_SET_RENDERS,
            labelled=bool(r.labels), multiplier=len(r.preps))

    try:
        retained, _dump_n, selection, cells = plan(visuals)
    except visual_budget.VisualBudgetError as exc:
        # already scored: never throw the scores away over a panel count.
        print(f"[render_wrapper] {V}: visuals=all refused ({exc}); "
              "falling back to visuals=auto — the scores are kept")
        visuals = "auto"
        retained, _dump_n, selection, cells = plan("auto")
    render_cands = [rec["_cand"] for rec in retained]
    # visuals="none" still renders the winner: <stem>_best_<frame>.png IS the view's
    # output (the report points at it), not a panel. It suppresses the per-candidate
    # set and, via pool/panels.py, the sheet.
    if not render_cands:
        render_cands = [best_c]
    elif best_c not in render_cands:
        render_cands.insert(0, best_c)   # invariant: the winner is always panelled
    print(f"[render_wrapper] {V}: panelling {len(render_cands)} candidate(s) "
          f"x {len(r.preps)} frame(s) ({selection}"
          + (f"; {cells} ordered cell(s) occupied" if cells is not None else "")
          + ")")

    cand_rank = {c: i for i, (c, _) in enumerate(ranked)}
    cand_images = {}                     # {cand: {frame: image}}
    for prep in r.preps:
        render.place_match_camera(r.a, r.ctx.cam_obj, r.ctx.center, r.ctx.radius,
                                  prep.fs.intrinsics)
        for c in render_cands:
            img = r.render_panel(prep, c, cand_rank, best_c)
            cand_images.setdefault(c, {})[prep.fs.name] = img
    best_images = {name: r.best_image(name, img)
                   for name, img in cand_images[best_c].items()}
    r.sess.restore()

    # `passes` is a flat list here and a per-frame dict in the other mode, which is
    # why it is not part of report_args(): the two searches genuinely differ.
    _write_report(r.a.out, r.STEM, V, r.cfg.mode, r.preps, r.ref_name, ranked,
                  best_c, best, best_images, cand_images=cand_images,
                  passes=r.passes, **r.report_args())


# --------------------------------------------------------------------------- #
# PER-FRAME mode: N independent fits, one rotation per frame
# --------------------------------------------------------------------------- #
def _run_per_frame(r, rast):
    """Run N independent sweeps — one rotation per frame, each frame its own argmax
    refined into its OWN window.

    There is no shared basin to exploit and no mean to rank by: a candidate's mean
    would average over whichever frames happened to visit it, since a refine pass
    scores a DIFFERENT candidate per frame. So this mode keeps the RAW
    `raw[cand][frame] = rec` matrix and selects per frame, exactly as N calls to
    `sweep` would."""
    V = r.V

    def merge_scored(scored, by_cand):
        """Accumulate raw (candidate, frame) recs across passes."""
        for c, frames in by_cand.items():
            scored.setdefault(c, {}).update(frames)
            r.prog.update(r.prog.best)
        return scored

    # ---- coarse pass + a refine window PER FRAME ---------------------------- #
    def n_for(frame):
        return sum(1 for recs in raw.values() if frame in recs)

    raw = merge_scored({}, r.score_pass(r.same, rast))
    windows = {p.fs.name: r.search for p in r.preps}
    for prep in r.preps:
        r.record_coarse(n_for(prep.fs.name), frame=prep.fs.name)
    for p in range(r.refine_n):
        if r.prog.expired():
            break
        nxt, centers, before = {}, {}, {}
        for prep in r.preps:
            bc = _best_for_frame(raw, prep, r.depth_weight)
            if bc is None:
                continue
            name = prep.fs.name
            windows[name], nxt[name] = r.refine_window(windows[name], bc)
            centers[name], before[name] = bc, n_for(name)
        if not nxt:
            break
        print(f"[render_wrapper] {V}: refine pass {p + 1}/{r.cfg.refine} "
              f"per frame ({len(nxt)} frame(s), "
              f"{sum(len(v) for v in nxt.values())} renders)")
        merge_scored(raw, r.score_pass(nxt, rast))
        # per frame, because each frame descended into its OWN basin: the windows
        # diverge after pass 1, and a single shared record would describe none of them.
        for name in nxt:
            r.record_refine(windows[name], n_for(name) - before[name],
                            centers[name], frame=name)

    r.prog.done()
    r.sess.end_scoring()
    r.timed_out = r.prog.expired()

    # per-frame winners; a frame with nothing scored (timeout mid-pass) drops out of
    # the report entirely rather than borrowing another frame's rotation.
    winners = {}
    for prep in r.preps:
        bc = _best_for_frame(raw, prep, r.depth_weight)
        if bc is not None:
            winners[prep.fs.name] = bc
    r.preps = [p for p in r.preps if p.fs.name in winners]
    if not r.preps:
        r.nothing_scored()
        return

    visuals, waived, canon_kinds = r.begin_panels()
    frame_ranked = {}
    for prep in r.preps:
        mine = [(c, recs[prep.fs.name]) for c, recs in raw.items()
                if prep.fs.name in recs]
        mine.sort(key=lambda cr: r.score_of(cr[1], prep), reverse=True)
        frame_ranked[prep.fs.name] = mine

    # ---- per-frame panels: plan per frame, ONE budget check for the order ---- #
    # Each frame is its own unit (multiplier=1), so a naive per-frame check would
    # wave 50 candidates x 10 frames through as ten innocent 50s. The counts are
    # summed and checked once — one honest message about the order's real size.
    def plan_per_frame(level):
        out, total, sel, cel = {}, 0, None, None
        for prep in r.preps:
            recs = [{"order": _canon_dict(c), "_cand": c}
                    for c, _ in frame_ranked[prep.fs.name]]
            retained, _dn, sel, cel = metrics.panel_plan(
                recs, canon_kinds, r.ranges, r.cfg.dump_topk, visuals=level,
                waived=waived, what=V, set_limit=_MAX_SET_RENDERS,
                labelled=bool(r.labels), multiplier=1, defer_budget=True)
            cands = [rec["_cand"] for rec in retained]
            w = winners[prep.fs.name]
            if not cands:
                cands = [w]              # visuals=none still renders the winner
            elif w not in cands:
                cands.insert(0, w)       # the winner is always panelled
            out[prep.fs.name] = cands
            total += len(cands)
        return out, total, sel, cel

    frame_cands, total_panels, selection, cells = plan_per_frame(visuals)
    if visuals == "all":
        try:
            visual_budget.check_budget(
                total_panels, waived=waived, what=f"{V} visuals=all",
                detail=f"summed over {len(r.preps)} frames, each scoring its own "
                       "candidates")
        except visual_budget.VisualBudgetError as exc:
            print(f"[render_wrapper] {V}: visuals=all refused ({exc}); "
                  "falling back to visuals=auto — the scores are kept")
            visuals = "auto"
            frame_cands, total_panels, selection, cells = plan_per_frame("auto")
    print(f"[render_wrapper] {V}: panelling {total_panels} image(s) across "
          f"{len(r.preps)} frame(s), one rotation each ({selection}"
          + (f"; up to {cells} ordered cell(s) occupied per frame"
             if cells is not None else "") + ")")

    rows, best_images = [], {}
    for prep in r.preps:
        render.place_match_camera(r.a, r.ctx.cam_obj, r.ctx.center, r.ctx.radius,
                                  prep.fs.intrinsics)
        name = prep.fs.name
        rank_of = {c: i for i, (c, _) in enumerate(frame_ranked[name])}
        w = winners[name]
        for c in frame_cands[name]:
            img = r.render_panel(prep, c, rank_of, w)
            rows.append({"frame": name, "cand": c, "rank": rank_of.get(c, 0),
                         "image": img})
            if c == w:
                best_images[name] = r.best_image(name, img)
    r.sess.restore()

    _write_per_frame_report(r.a.out, r.STEM, V, r.cfg.mode, r.preps, r.ref_name,
                            raw, winners, rows, best_images,
                            frame_passes=r.frame_passes, **r.report_args())


# --------------------------------------------------------------------------- #
# report + paste writer (reuses metrics pure helpers)
# --------------------------------------------------------------------------- #
def panel_image_name(stem_prefix, cand, frame_stem, labels, cand_rank, best_c):
    """The panel filename for one candidate x one frame — UNIQUE per candidate.

    A labelled set (oapply, --sweep-candidates) names by label. Grid mode panels
    several binned cells, so the runners-up take a score-rank suffix and only the winner
    keeps the bare `_best_` name the report points at."""
    if labels:
        return f"{stem_prefix}_{labels[cand]}_{frame_stem}.png"
    if cand == best_c:
        return f"{stem_prefix}_best_{frame_stem}.png"
    return f"{stem_prefix}_top_{cand_rank.get(cand, 0):02d}_{frame_stem}.png"
