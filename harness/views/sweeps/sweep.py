"""sweep view — search a frame's POSE and/or JOINT states for maximum silhouette
IoU (minus a depth penalty, with --tracking) under the fixed camera 0.

This is the SINGLE unified sweep view. It searches a UNIFIED DOF vector — any mix
of camera-frame pose "ORDER" increments (roll/yaw/pitch/dpx/dpy/tz — see
rig/lie.apply_order) and absolute JOINT states — so one view covers pose-only,
joint-only, and coupled pose+joint fits:

  * name only pose DOFs        -> a pose fit;
  * name only joints           -> an articulation fit;
  * name both / --sweep-space both
                               -> a COUPLED fit, a grid over the joint↔pose DOFs
                                  jointly.

Candidates come from bounded differential evolution by default. Optional Powell
polishing is explicit; a Cartesian grid with coarse-to-fine refinement remains
available for dense landscape inspection.

The view itself is THIN: it resolves the knobs off ctx.args into a SweepConfig and
hands off to the layered blocks under views/sweeps/ —
  dof_space   : the umbrella (a DOF vector -> a posed candidate);
  planner : search bounds and grid generation;
  search_policy : adaptive numerical optimization;
  engine  : the dispatcher (the Blender render/score loop + winner re-render);
  metrics : score_of, landscape/plateau stats, the 2-D field, the report.

Efficiency comes from staying in ONE Blender process (EEVEE + transparent film +
persistent BVH), scoring at a low resolution (1 AA sample, long side capped); the
winner is re-rendered at full match-res. Non-destructive: the base pose + render
resolution / intrinsics are restored, so match / turntable / pose.json / the GLB
are unaffected.
"""

from views.sweeps.lib import engine


def _cfg_from_args(a):
    """Build a SweepConfig from ctx.args (flags declared in rig/args.py)."""
    return engine.SweepConfig(
        mask=a.sweep_mask,
        hand_mask=a.sweep_hand_mask,
        hand_dilate=a.sweep_hand_dilate,
        quality=a.sweep_quality,
        ranges=a.sweep_ranges,
        space=a.sweep_space,
        refine=a.sweep_refine,
        refine_shrink=a.sweep_refine_shrink,
        topk=a.sweep_topk,
        depth_weight=a.sweep_depth_weight,
        timeout=a.sweep_timeout,
        pose_start=a.sweep_pose_start,
        start_shift=a.sweep_start_shift,
        candidates=a.sweep_candidates,
        grid_slice=a.sweep_grid_slice,
        dump_topk=a.sweep_dump_topk,
        joints=a.sweep_joints,
        angle_preset=a.sweep_angle_preset,
        visuals=a.sweep_visuals,
        waive_visual_budget=a.waive_visual_budget,
        strategy=a.sweep_strategy,
        de_max_evals=a.sweep_de_max_evals,
        de_popsize=a.sweep_de_popsize,
        de_seed=a.sweep_de_seed,
        de_mutation=a.sweep_de_mutation,
        de_recombination=a.sweep_de_recombination,
        polish=a.sweep_polish,
        polish_max_evals=a.sweep_polish_max_evals,
    )


def render_view(ctx):
    engine.run(ctx, _cfg_from_args(ctx.args))
