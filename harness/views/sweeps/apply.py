"""apply view — the IMPERATIVE / directional verb: "tilt 90 degrees. Boom. render."

Where `sweep` SEARCHES (score a whole grid of candidates against a mask, rank, pick
the best), `apply` DOES ONE THING: it composes a directional ORDER you name — any mix
of camera-frame pose increments (roll/yaw/pitch/dpx/dpy/tz, see
rig/lie.apply_order) and/or absolute joint states — onto the frame's current pose,
renders it, and **scores that one config** against the mask (IoU, + Pi3X depth with
--tracking) so you can glance at the render and read "yep, decent IoU". It writes a
paste-ready pose/joints block to copy into scene.py FRAMES.

Maximal reuse: `apply` is literally a `sweep` of exactly ONE candidate. This view is a
thin front-end — it turns `--apply 'yaw:90;door:70'` into a degenerate one-point
grid (`yaw:90,90,1;door:70,70,1`, no refine) and hands a SweepConfig to
`sweeps/engine.run`, which owns the mask/IoU/depth scoring, the full-res winner
re-render, and the report. `out_stem="apply"` names the outputs apply_best.png /
apply.json / apply.txt (and labels the legend/logs) so they never collide with a
co-requested sweep. Everything is scored/rendered/restored by the same non-destructive
engine, so match/turntable/pose.json/GLB are unaffected — the durable channel is the
paste block.

It reuses `sweep`'s scoring flags: --sweep-mask (required), --sweep-hand-mask,
--sweep-hand-dilate, --sweep-depth-weight (with --tracking), --sweep-quality, --sweep-pose-start.
"""

from views.sweeps.lib import engine, planner


def _cfg_from_args(a):
    """A SweepConfig that scores the single applied candidate, writing apply.* outputs."""
    return engine.SweepConfig(
        mask=a.sweep_mask,
        hand_mask=a.sweep_hand_mask,
        hand_dilate=a.sweep_hand_dilate,
        quality=a.sweep_quality,
        ranges=planner.apply_ranges_spec(a.apply),
        space=a.sweep_space,
        refine=0,                    # imperative: no coarse->fine search
        topk=1,                      # a single applied config, no alternatives
        depth_weight=a.sweep_depth_weight,
        timeout=a.sweep_timeout,
        pose_start=a.sweep_pose_start,
        dump_topk=0,
        joints=a.sweep_joints,
        out_stem="apply",
        view="apply",
        visuals=a.sweep_visuals,
        waive_visual_budget=a.waive_visual_budget,
    )


def render_view(ctx):
    a = ctx.args
    if not a.apply.strip():
        print("[render_wrapper] apply: --apply 'dof:value;...' is required; "
              "skipping apply")
        return
    engine.run(ctx, _cfg_from_args(ctx.args))
