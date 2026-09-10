"""oapply view — the PER-FRAME imperative object-centric verb: "try these turns on
each of these frames; tell me which one each frame wants."

ONE rotation PER FRAME, named. This is `apply`'s object-centric twin: where `apply`
composes ONE camera-frame order onto ONE frame, `oapply` applies a named set of
CANONICAL rotations (rx/ry/rz, degrees about the object's own +X/+Y/+Z, right-
multiplied about the frame's own FK-AABB centre: M_k' = M_k @ T(c_k)·R_extra·T(-c_k),
so the object spins in place on screen) to each of N frames INDEPENDENTLY, and every
frame picks its own winner by its own gate IoU. Semantically it is N `apply` runs that
happen to share one Blender process and one candidate set — nothing is aggregated,
nothing is inherited.

Reach for it when the frames disagree with each other: a window seam where some
frames sit in a flipped basin, "this one came out facing backwards". The report's
per-frame rotation column then says whether they converged. When the object itself
was built mis-oriented — wrong in EVERY frame, one decision to commit — the verb is
`oapply_all` instead.

`--oapply` names the candidate SET ('|'-separated; see planner.parse_canon_candidates):
'rz:180', 'flip:z', or the symmetry PANEL 'flips' (identity + a 180deg flip about each
canonical axis). `--opreset 'z:quarters'` names an axis plus a range of angles.
Every candidate is rendered for every frame (oapply_<label>_<frame>.png), because flip
twins tie on IoU and the decision is made by LOOKING.

Thin front-end: it parses the flags into a labelled candidate list and hands a
per-frame SharedConfig to `shared_engine.run`, which owns the render/score matrix and
the report.
"""

from views.sweeps.lib import planner, shared_engine


def _cfg_from_args(a):
    return shared_engine.SharedConfig(
        mode=shared_engine.PER_FRAME,   # one rotation PER frame, chosen per frame
        masks_dir=a.masks_dir,
        hand_masks_dir=a.hand_masks_dir,
        hand_dilate=a.sweep_hand_dilate,
        quality=a.sweep_quality,
        candidates=planner.imperative_candidates(a.oapply, a.opreset,
                                                 a.sweep_quality),
        refine=0,                    # imperative: no coarse->fine search
        depth_weight=a.sweep_depth_weight,
        timeout=a.sweep_timeout,
        out_stem="oapply",
        view="oapply",
        visuals=a.sweep_visuals,
        waive_visual_budget=a.waive_visual_budget,
    )


def render_view(ctx):
    if not shared_engine.require_rotations(ctx.args, "oapply"):
        return
    shared_engine.run(ctx, _cfg_from_args(ctx.args))
