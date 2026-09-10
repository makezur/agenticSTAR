"""oapply_all view — the WHOLE-RUN imperative verb: "flip it 180 everywhere. render."

ONE shared canonical rotation for EVERY frame. Where `osweep` SEARCHES a rotation
per frame and `oapply` applies named ones per frame, `oapply_all` applies the ONE(S)
you name to the whole run — each candidate a canonical rotation vector (rx/ry/rz,
degrees about object +X/+Y/+Z) right-multiplied onto EVERY frame's pose jointly
(M' = M @ R_extra — about the canonical ORIGIN, unlike the per-frame verbs' FK-AABB
pivot, because one shared right-factor is exactly the correction a rotated build()
would be and is what lets seeded frames inherit the reference's paste) — then
renders + scores all frames against their masks so you can glance at the per-frame
IoU table and read "yep, that un-flipped it."

This is the verb for a BUILD-ORIENTATION error: the object came out mis-oriented, so
it reads wrong in every frame, and the fix is ONE decision committed once. When the
frames disagree with each other instead, the honest verb is the per-frame `oapply`,
which lets each frame pick its own basin.

`--oapply` names a candidate SET ('|'-separated; see planner.parse_canon_candidates):
one custom rotation ('rz:180'), a symmetry flip ('flip:z'), or the default
symmetry PANEL ('flips' = identity + the 180deg flip about each canonical axis) for
the seam-flip check on near-symmetric objects. `--opreset 'z:quarters'` names an axis
plus a range of angles instead. Small sets re-render EVERY candidate per frame at full
match-res (oapply_all_<label>_<frame>.png) — flip twins tie on IoU, so the decision is
made by LOOKING, and the report's candidate matrix says which frames prefer which
basin.

Maximal reuse: `oapply_all` is literally an `osweep` of the named candidates in
SHARED mode. This view is a thin front-end — it parses `--oapply`/`--opreset` into an
explicit candidate list and hands a SharedConfig to `shared_engine.run`, which owns
the multi-frame render/score and the report. `out_stem="oapply_all"` names the outputs
oapply_all_best_<frame>.png / oapply_all.json / oapply_all.txt so they never collide
with a co-requested per-frame oapply or osweep.
"""

from views.sweeps.lib import planner, shared_engine


def _cfg_from_args(a):
    return shared_engine.SharedConfig(
        mode=shared_engine.SHARED,   # ONE rotation for the whole run
        masks_dir=a.masks_dir,
        hand_masks_dir=a.hand_masks_dir,
        hand_dilate=a.sweep_hand_dilate,
        quality=a.sweep_quality,
        # the named candidate set bypasses the grid (and refine) entirely —
        # oapply_all is an osweep of exactly these candidates.
        candidates=planner.imperative_candidates(a.oapply, a.opreset,
                                                 a.sweep_quality),
        refine=0,                    # imperative: no coarse->fine search
        depth_weight=a.sweep_depth_weight,
        timeout=a.sweep_timeout,
        out_stem="oapply_all",
        view="oapply_all",
        visuals=a.sweep_visuals,
        waive_visual_budget=a.waive_visual_budget,
    )


def render_view(ctx):
    if not shared_engine.require_rotations(ctx.args, "oapply_all"):
        return
    shared_engine.run(ctx, _cfg_from_args(ctx.args))
