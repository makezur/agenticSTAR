"""osweep view — SEARCH a canonical object rotation PER FRAME.

The object-centric sibling of `sweep`. Where `sweep` searches CAMERA-frame pose
increments left-multiplied onto ONE frame, `osweep` searches a rotation in the
object's own CANONICAL frame — the rotation vector (rx, ry, rz), degrees about
object +X/+Y/+Z — right-multiplied onto that frame's pose about the frame's own
FK-AABB centre (M' = M @ T(c)·R_extra·T(-c); the same material point `sweep`
orbits, so the object spins in place on screen), and it does so for each of N
frames INDEPENDENTLY: every frame gets its own grid search, its own argmax by its
own gate IoU, its own refine window. Semantically it is N `sweep`
runs that happen to share one Blender process and one candidate grid — nothing is
aggregated, nothing is inherited.

Use it when a frame reads wrong in a way that is easier to name in the object's own
axes ("this one came out facing backwards") than as a camera-frame yaw, which is the
correction `sweep` cannot express: a canonical turn about the object's up-axis is not
a camera-frame yaw except for an upright object. When the OBJECT was built
mis-oriented — wrong in every frame, one decision to commit — the verb is
`oapply_all` (with `--oapply 'flips'` or `--opreset 'z:full'`) instead; there is
deliberately no whole-run *searched* mode, because ranking a shared rotation by MEAN
IoU can pick one that is mediocre everywhere.

Thin front-end: resolve the knobs off ctx.args into a SharedConfig and hand off to
`shared_engine.run` (the frame x candidate scoring loop + report). Its imperative
sibling is the per-frame `oapply` view (named rotations instead of a search).
"""

from views.sweeps.lib import shared_engine


def _cfg_from_args(a):
    return shared_engine.SharedConfig(
        mode=shared_engine.PER_FRAME,   # one searched rotation PER frame
        masks_dir=a.masks_dir,
        hand_masks_dir=a.hand_masks_dir,
        hand_dilate=a.sweep_hand_dilate,
        quality=a.sweep_quality,
        ranges=a.osweep_ranges,
        refine=a.sweep_refine,
        refine_shrink=a.sweep_refine_shrink,
        depth_weight=a.sweep_depth_weight,
        timeout=a.sweep_timeout,
        angle_preset=a.osweep_angle_preset,
        preset=a.opreset,
        out_stem="osweep",
        view="osweep",
        visuals=a.sweep_visuals,
        waive_visual_budget=a.waive_visual_budget,
        dump_topk=a.sweep_dump_topk,
    )


def render_view(ctx):
    shared_engine.run(ctx, _cfg_from_args(ctx.args))
