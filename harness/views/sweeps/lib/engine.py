"""engine.py — the sweep DISPATCHER: given ranges, render + score + compute stats.

This is the only Blender-side module of the sweep machinery. It:
  1. sets up the low-res scoring camera, masks, and (with --tracking) depth supervision;
  2. builds the fixed base pose `P0` + `OrderCtx` and the unified DOF `Space`;
  3. wraps `space.realize` + `scene.pose_frame` + render + IoU/depth score into a
     `score(x)` closure over one persistent Blender process;
  4. scores the planner's grid product (with optional coarse->fine refinement)
     through that closure;
  5. ranks, restores the scene, re-renders the winner (+ optional contact sheet),
     computes the landscape stats / 2-D field, and writes the report.

Everything that is pose/joint/coupled-specific lives in `dof_space`; everything about
generating candidates lives in `planner`; everything about scoring math + the
report lives in `metrics`. This module is the glue that runs them against Blender.
"""

import os

import bpy
import numpy as np
from mathutils import Vector

from rig import imaging, lie, render, scene, transforms
from rig.transforms import _mat_to_np
from core import cam_math, centre_calculation, visual_budget
from core import depth_config
from core import joints as joints_core
from views import depth as depth_view
from views.sweeps.lib import dof_space, metrics, pivot, planner, search_policy
# the config is a pure dataclass in config; re-exported here because every caller
# (views/sweeps/sweep.py, views/apply.py) reaches it as engine.SweepConfig.
from views.sweeps.lib.config import SweepConfig       # noqa: F401
from views.sweeps.lib.runtime import (
    base_scale as _base_scale,
    DepthSupervision as _DepthSupervision,
    make_deadline as _deadline,
    Progress as _Progress,
    resolve_depth_weight as _resolve_depth_weight,
    ScoringSession as _ScoringSession)

# behind-camera sentinel: a score so low the ranking never picks it.
NEG = -1e18


# --------------------------------------------------------------------------- #
# space construction (Blender-side: P0, OrderCtx, joint limits)
# --------------------------------------------------------------------------- #
def _resolve_sweep_space(declared, ranges):
    """The resolved 'pose'|'joint'|'both' for this run.

    A DECLARED space wins — either --sweep-space, or the one `run` infers from an
    explicit candidate list. Otherwise infer from the named DOFs: pose-order names
    -> pose, joint names -> joint, a mix -> both; an EMPTY ranges spec with no
    joints declared -> pose (back-compat). With joints declared and
    an empty spec, still default to pose (the historical `sweep` default) — the
    agent asks for joints explicitly by naming them or via --sweep-space.

    Takes the two values rather than the config: the resolved space is a LOCAL of
    the run, not a field the engine writes back onto the caller's config.
    """
    if declared in ("pose", "joint", "both"):
        return declared
    names = _named_dofs(ranges)
    if not names:
        return "pose"
    has_pose = any(dof_space.classify_dof(n) == dof_space.POSE for n in names)
    has_joint = any(dof_space.classify_dof(n) == dof_space.JOINT for n in names)
    if has_pose and has_joint:
        return "both"
    return "joint" if has_joint else "pose"


def _named_dofs(spec):
    """The DOF names in a --sweep-ranges/-steps spec (before validation)."""
    out = []
    for part in (spec or "").split(";"):
        part = part.strip()
        if part and ":" in part:
            out.append(part.split(":", 1)[0].strip())
    return out


def _sweepable_joint_defs(joint_defs, cfg, sweep_space):
    """Joint defs relevant to this space, honoring --sweep-joints selector for the
    empty-spec default. Pose-only space -> none. Named-DOF spec -> only the named
    joints matter (validated in the space). Empty spec + joint/both -> all declared
    non-fixed joints (optionally filtered by --sweep-joints)."""
    if sweep_space == "pose":
        return []
    by_name = {j.get("name"): j for j in joint_defs if j.get("name")}
    named = [n for n in _named_dofs(cfg.ranges)
             if dof_space.classify_dof(n) == dof_space.JOINT]
    if named:
        for n in named:
            if n not in by_name:
                raise ValueError(
                    f"--sweep-ranges: unknown joint {n!r} "
                    f"(declared: {', '.join(by_name) or 'none'})"
                    + planner.canon_cross_hint(n))
        return [by_name[n] for n in named]
    # empty spec -> defaults over all sweepable joints (optional selector)
    wanted = None
    if cfg.joints.strip():
        wanted = [x.strip() for x in cfg.joints.split(",") if x.strip()]
        for w in wanted:
            if w not in by_name:
                raise ValueError(f"--sweep-joints: unknown joint {w!r} "
                                 f"(declared: {', '.join(by_name) or 'none'})"
                                 + planner.canon_cross_hint(w))
    out = []
    for name, j in by_name.items():
        if wanted is not None and name not in wanted:
            continue
        if j.get("type", "fixed") == "fixed":
            continue
        out.append(j)
    return out


def _named_pose_dofs(cfg, sweep_space):
    """Pose-order DOF names to include in the space. A named spec uses exactly the
    named pose DOFs; an empty spec uses the space's default box keys."""
    if sweep_space == "joint":
        return []
    named = [n for n in _named_dofs(cfg.ranges)
             if dof_space.classify_dof(n) == dof_space.POSE]
    if named:
        # validate against the reserved set
        for n in named:
            if n not in dof_space.POSE_DOF_SET:
                raise ValueError(f"--sweep-ranges: {n!r} is not a pose DOF")
        return named
    if sweep_space == "both":
        return list(planner.POSE_COUPLED_DEFAULT.keys())
    return list(dof_space.POSE_DOFS)


def _make_order_ctx(P0, joint_defs, joint_states, canonical, objs, place, sw, sh):
    """The OrderCtx (pivot centre + object depth + focal at scoring res) + the
    pivot's wire record, for the object posed at `P0`.

    THE PIVOT IS A MATERIAL POINT, MEASURED AT THE FRAME'S ARTICULATION. It is the
    canonical-frame AABB centre at `joint_states` (core.centre_calculation), carried
    into the camera by the frame's own placement acting on that point
    (`lie.pose_apply`, i.e. `s*R@c_canon + t`). A union AABB taken in POSED space
    is not rotation-equivariant across multiple parts, so measuring in the canonical
    frame and transporting the point is what keeps the pivot on one MATERIAL point.

    `joint_states` is the frame's BASE articulation, resolved once per run. A coupled
    `--sweep-space both` run varies joints per candidate, but the pivot stays frozen
    here on purpose: `dof_space.realize` composes every candidate onto a FIXED P0 and
    OrderCtx so a pasted winner reproduces its scored render bit-for-bit, and a
    per-candidate pivot would break both that and the single serialized pivot the
    report records."""
    # the measurement is shared with the canonical grammar (lib/pivot.py: same
    # material point for both); the CARRY is this grammar's own — the placement
    # acting on a point (s*R@c + t), which lie owns.
    c_canon, _ = pivot.frame_pivot(joint_defs, joint_states, canonical, objs)
    centre = lie.pose_apply(P0, c_canon)
    intr = place.get("intr_effective")
    if intr is not None:
        fx, fy = float(intr["fx"]), float(intr["fy"])
    else:
        import math
        f = cam_math.focal_from_fov(
            math.degrees(bpy.context.scene.camera.data.angle), max(sw, sh))
        fx = fy = f
    ctx = dof_space.OrderCtx(centre=centre, depth_z=float(centre[2]), fx=fx, fy=fy,
                             width=float(sw), height=float(sh))
    return ctx, centre_calculation.pivot_record(centre, c_canon, joint_states)


def build_space(pose_dof_names, space_joint_defs, start_placement, base_states,
                canonical, objs, place, width, height, joint_defs=None):
    """Construct the unified DOF `Space` for one frame — the single seam shared by
    the `run()` search and the imperative `apply` view.

    Builds the fixed base pose `P0`, the `OrderCtx` (pivot/depth/focal, only if any
    pose DOFs are named), the swept joints' limits, and folds in the held base joint
    states. `place`/`width`/`height` fix the OrderCtx geometry at the CALLER's render
    resolution; the pose-order DOF units are resolution-invariant (dpx/dpy are
    frame-fractions, tz a depth-fraction — see rig/lie.apply_order), so a `Space`
    built at full match res and one built at the low scoring res agree for the same
    DOF vector. A joint-only space carries `order_ctx=None` (the base pose is P0).

    `joint_defs` is ALL the scene's joints, not just the swept ones: the pivot is
    measured on the object's actual articulated shape, so every joint that bends the
    geometry counts — including ones this space holds fixed. It defaults to the swept
    defs only so an existing caller keeps working; the engine passes the full set.

    Returns `(space, pivot_rec)` — `pivot_rec` is the wire record of where the
    order increments orbit (`core.centre_calculation.pivot_record`), or None for a
    joint-only space, which has no order geometry to record. (Named `pivot_rec`,
    not `pivot`: that is the `lib.pivot` MODULE here, and a local `pivot` would
    shadow it for any future measurement call inside this function.)
    """
    base_scale_val = start_placement["scale"]
    order_ctx = None
    pivot_rec = None
    P0 = lie.pose_from_matrix(_mat_to_np(
        transforms.compose_pose(**start_placement)))
    if pose_dof_names:
        order_ctx, pivot_rec = _make_order_ctx(
            P0, joint_defs if joint_defs is not None else space_joint_defs,
            base_states, canonical, objs, place, width, height)
    joint_limits = {j.get("name"): transforms.joint_limit(j)
                    for j in space_joint_defs}
    joint_types = {j.get("name"): j.get("type", "fixed")
                   for j in space_joint_defs}
    dof_names = list(pose_dof_names) + [j.get("name") for j in space_joint_defs]
    kinds = {d: dof_space.classify_dof(d) for d in dof_names}
    space = dof_space.Space(dof_names=dof_names, kinds=kinds, P0=P0,
                        order_ctx=order_ctx, base_scale=base_scale_val,
                        base_states=base_states, joint_limits=joint_limits,
                        joint_types=joint_types)
    return space, pivot_rec


# the pivot's wire rounding lives in metrics (bpy-free) because BOTH report
# families now serialize pivot records: this grammar's camera-frame one and the
# canonical grammar's per-frame ones (shared_report is bpy-free by contract and
# cannot import this module).
_pivot_out = metrics.pivot_out


def _placement_matrix(placement):
    """The 4x4 for a placement, composed from its authoritative
    quaternion/translation/scale fields. Those fields are authoritative and
    full-precision, so recomposing is exact and we keep NO derived matrix cache —
    a cached 4x4 is one more thing that can disagree with the placement it came
    from, and the placement is what gets pasted into scene.py."""
    return transforms.compose_pose(**transforms.normalize_placement(placement))


def _canonical_corners(canonical, objs):
    """All parts' bbox corners at the canonical pose, as one (4, 8N) homogeneous
    array — fixed for a whole sweep (the loop only changes the placement M), so
    the per-candidate behind-camera check is a single 4x4 matmul instead of a
    Python loop over every corner of every part (~0.55ms -> ~0.02ms/candidate)."""
    cols = []
    for o in objs:
        C = canonical[o]
        for corner in o.bound_box:
            p = C @ Vector(corner)
            cols.append((p.x, p.y, p.z, 1.0))
    return np.array(cols, dtype=np.float64).T


def _behind_camera(cam_obj, M, corners):
    """True if the posed object bbox centre sits at/behind the camera plane.
    `corners` is the (4, 8N) _canonical_corners array; M the candidate 4x4."""
    from bpy_extras.object_utils import world_to_camera_view
    pts = _mat_to_np(M) @ corners
    center = 0.5 * (pts[:3].min(axis=1) + pts[:3].max(axis=1))
    ndc = world_to_camera_view(bpy.context.scene, cam_obj, Vector(center))
    return ndc.z <= 0.0


# --------------------------------------------------------------------------- #
# preparation
# --------------------------------------------------------------------------- #
# `run` was one 410-line function: five skip paths, the whole search-space setup, the
# scoring loop, refinement, the winner re-render and the report, in one scope with
# ~25 live locals. The phases were already marked with comment banners — this makes
# them callable, so each can be read (and the skip paths found) without holding the
# other four in your head. `_prepare` owns everything up to the first render and
# hands over ONE `_Prep`; the canonical engine next door already works this way
# (`shared_engine._prepare` -> `_Run`), so the two engines now read alike.
class _Prep:
    """Everything the scoring loop and the report need, resolved before the first
    render.

    A plain attribute bag rather than a tuple: the setup produces ~25 values, and a
    positional handoff is where a refactor like this goes wrong."""

    def __init__(self, ctx, cfg, sess, objs, canonical, joint_defs,
                 all_joint_names, start_placement, base_states, explicit_candidates,
                 space, sweep_space, ranges, quality, gate_field, grid_iou,
                 depth_sup, depth_weight, deadline, sl, sw, sh, pivot=None):
        self.ctx, self.cfg, self.sess = ctx, cfg, sess
        self.a = ctx.args
        self.V, self.STEM = cfg.view, cfg.out_stem
        self.objs, self.canonical, self.joint_defs = objs, canonical, joint_defs
        self.all_joint_names = all_joint_names
        self.start_placement, self.base_states = start_placement, base_states
        self.explicit_candidates = explicit_candidates
        self.space, self.sweep_space, self.ranges = space, sweep_space, ranges
        self.dof_names, self.kinds = space.dof_names, space.kinds
        self.quality = quality
        self.gate_field, self.grid_iou = gate_field, grid_iou
        self.depth_sup, self.depth_weight = depth_sup, depth_weight
        self.deadline, self.sl = deadline, sl
        self.sw, self.sh = sw, sh
        # where the order increments orbit (None for a joint-only space, which has
        # no order geometry). Serialized by `_write_the_report`.
        self.pivot = pivot


def _prepare(ctx, cfg):
    """Resolve the space, masks, camera and depth supervision -> (_Prep, rast).

    Returns `(None, None)` when the run cannot proceed; every one of those paths has
    already PRINTED its own named reason (a missing mask, no posed frame, no mesh
    parts, no sweepable joints, no DOFs), because "skipping sweep" with no cause is
    the report an agent cannot act on."""
    a = ctx.args
    V, STEM = cfg.view, cfg.out_stem
    strategy = str(cfg.strategy).lower()
    if strategy not in ("grid", "de"):
        raise ValueError("--sweep-strategy must be grid or de")
    if strategy == "de" and cfg.grid_slice and not cfg.candidates:
        raise ValueError("--sweep-grid-slice applies only to grid search")
    if not cfg.mask or not os.path.isfile(cfg.mask):
        print(f"[render_wrapper] {V}: --sweep-mask <object mask png> is required "
              f"(got {cfg.mask!r}); skipping {V}")
        return None, None
    if ctx.frame is None:
        print(f"[render_wrapper] {V}: no frame posed; skipping")
        return None, None

    objs = scene.mesh_parts()
    if not objs:
        print(f"[render_wrapper] {V}: no mesh parts to pose; skipping")
        return None, None

    joint_defs = ctx.spec.joint_defs if ctx.spec is not None else []
    if cfg.start_shift and cfg.candidates:
        raise ValueError("--sweep-start-shift and --sweep-candidates are mutually "
                         "exclusive: explicit candidates carry absolute pose/joints, "
                         "so a directional shift onto them is ill-defined")
    explicit_items = planner.parse_candidates(cfg.candidates)
    # an explicit candidate list carries its own pose/joints, so it DECLARES the
    # space. Resolved into a local -- the engine never writes back onto cfg, so one
    # config stays reusable across runs (which is what a pooled worker does).
    declared_space = cfg.space
    if explicit_items and not declared_space:
        has_joint = any(bool(c.get("joints")) for c in explicit_items)
        has_pose = any(c.get("pose") is not None for c in explicit_items)
        declared_space = ("both" if (has_joint and has_pose) else
                          "joint" if has_joint else "pose")
    sweep_space = _resolve_sweep_space(declared_space, cfg.ranges)
    space_joint_defs = _sweepable_joint_defs(joint_defs, cfg, sweep_space)
    if sweep_space in ("joint", "both") and not space_joint_defs:
        print(f"[render_wrapper] {V}: no sweepable joints (all fixed / none "
              "selected / scene declares none); use a pose sweep instead. Skipping")
        return None, None
    pose_dof_names = _named_pose_dofs(cfg, sweep_space)

    canonical = ctx.canonical or scene.capture_canonical(objs)
    quality = render.clamp_quality(cfg.quality)
    sess = _ScoringSession(ctx.cam_obj, objs, a, quality, a.out, STEM, view=V)

    # WHERE THE GRID STARTS (its centre / P0). --sweep-pose-start JSON wins, else the
    # frame's own placement.
    start_placement = transforms.normalize_placement(_resolve_pose_start(cfg, ctx))
    base_scale_val = start_placement["scale"]

    # held joint states (the frame's declared states; swept joints override).
    # No 0.0 fallback: a joint state is ABSOLUTE, so an unstated joint has no
    # defaultable value and filling one with 0.0 would seed the whole grid on a
    # STRAIGHTENED joint while the report listed the frame's real state. The
    # frame's states are guaranteed complete by the scene-load gate and by
    # serve.py's merge-not-replace, so a failure here means a programmatic caller
    # built a FrameSpec by hand — which is exactly what should be refused.
    all_joint_names = [j.get("name") for j in joint_defs if j.get("name")]
    frame_states = joints_core.require_complete_states(
        joint_defs, ctx.frame.joint_states or {},
        f"{V}: frame {getattr(ctx.frame, 'name', '?')!r}")
    base_states = {n: float(frame_states[n]) for n in all_joint_names
                   if n in frame_states}
    # `fixed` joints are exempt from the completeness rule (identity at any
    # state), so a scene may declare one with no state anywhere. Carry it at 0.0
    # so `realize` still emits a value for every declared joint — this is not a
    # silent default: a fixed joint has no DOF for the number to be wrong about.
    base_states.update({n: 0.0 for n in all_joint_names if n not in base_states})
    explicit_candidates = _realize_explicit_candidates(
        explicit_items, start_placement, base_states, base_scale_val, joint_defs
    )

    # masks
    mask_bool, keep = imaging.load_mask_grid(cfg.mask, cfg.hand_mask,
                                             cfg.hand_dilate)
    gate_field = "iou_visible" if keep is not None else "iou_raw"

    # low-res scoring camera
    sw, sh = sess.sw, sess.sh
    bpy.context.scene.render.resolution_x = sw
    bpy.context.scene.render.resolution_y = sh
    place = render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius,
                                      ctx.frame_intr)

    # --sweep-start-shift: compose a directional order onto the base BEFORE building the
    # search space, so the sweep hunts a window AROUND the shifted pose. Reuses the
    # `apply` grammar + `realize` seam, then rebuilds P0/OrderCtx on the shifted start
    # (byte-faithful to the apply -> --sweep-pose-start -> sweep two-call workflow).
    if cfg.start_shift:
        start_placement, base_states, base_scale_val = _apply_start_shift(
            cfg, start_placement, base_states, joint_defs, canonical, objs,
            place, sw, sh)
        print(f"[render_wrapper] {V}: start shifted by {cfg.start_shift!r}; searching "
              "around the shifted pose")

    # fixed P0 + OrderCtx (only if pose DOFs) + joint limits -> the unified Space,
    # built at the scoring resolution. Shared with the imperative `apply` view.
    space, pivot_rec = build_space(pose_dof_names, space_joint_defs,
                                   start_placement, base_states, canonical, objs,
                                   place, sw, sh, joint_defs=joint_defs)
    dof_names = space.dof_names
    if not dof_names and not explicit_candidates:
        print(f"[render_wrapper] {V}: no DOFs to search; skipping")
        sess.restore()
        return None, None

    # Parse either a lattice or continuous bounds against the same unified space.
    # In both cases P0/--sweep-pose-start is the basin's reference pose.
    angle_presets = planner.parse_angle_presets(cfg.angle_preset)
    ranges = ({} if explicit_candidates else
              (planner.parse_bounds(cfg.ranges, space, quality, sweep_space,
                                    angle_presets)
               if strategy == "de" else
               planner.parse_ranges(cfg.ranges, space, quality, sweep_space,
                                    angle_presets)))

    # a grid slice shards the grid across parallel runs (and disables refine)
    sl = (None if explicit_candidates or strategy == "de"
          else _parse_slice(cfg.grid_slice))

    deadline = _deadline(cfg.timeout)
    rast = sess.begin_scoring()
    # precomputed exact IoU (folds the per-candidate nearest-upsample onto the
    # mask grid into two dot products; identical numbers to iou_on_grid)
    grid_iou = imaging.GridIoU(mask_bool, keep, (sh, sw))

    # depth supervision (unchanged machinery). The per-run depth_config.json
    # "cost" flip switch (default on) can force it off regardless of the weight:
    # "cost": false -> depth_weight 0 -> pure-IoU ranking, no DepthSupervision.
    depth_weight = _resolve_depth_weight(cfg.depth_weight, a.out)
    depth_sup = None
    if depth_weight > 0.0:
        depth_sup = _DepthSupervision.maybe_create(
            a.tracking, getattr(ctx.frame, "view_index", None), mask_bool, keep,
            (sh, sw), _base_scale(base_scale_val), a.out, STEM)
    if depth_sup is not None:
        if rast is None:
            depth_sup.setup()   # compositor EXR rider (EEVEE path only; the
            # raster path reads camera-Z straight off the framebuffer)
        print(f"[render_wrapper] {V}: depth supervision ON "
              f"(combined = {gate_field} - {depth_weight:g} * depth_canon)")
    else:
        depth_weight = 0.0

    return _Prep(ctx, cfg, sess, objs, canonical, joint_defs, all_joint_names,
                 start_placement, base_states, explicit_candidates, space,
                 sweep_space, ranges, quality, gate_field, grid_iou, depth_sup,
                 depth_weight, deadline, sl, sw, sh, pivot=pivot_rec), rast


def run(ctx, cfg):
    """Score a sweep for one frame and write sweep.json / sweep.txt / sweep_best.png.

    One entry point for every space (pose / joint / coupled) AND for the imperative
    `apply` view (a single explicit candidate; see views/apply.py). `cfg.view` /
    `cfg.out_stem` label the logs and name the outputs. See the module docstring."""
    p, rast = _prepare(ctx, cfg)
    if p is None:
        return
    _search_and_report(p, rast)


def _resolved_topk(value, strategy, explicit_candidates=False):
    """Literal positive top-K, else the strategy default."""
    value = int(value)
    if value < 0:
        raise ValueError("--sweep-topk must be non-negative")
    if value:
        return value
    return 6 if strategy == "de" and not explicit_candidates else 10


def _split_budget(total, count, *, minimum, flag):
    """Split an integer total deterministically, with at most one call of skew."""
    total, count, minimum = int(total), int(count), int(minimum)
    if count <= 0:
        raise ValueError("--sweep-topk must be positive after default resolution")
    if total < minimum * count:
        raise ValueError(
            f"{flag}={total} is too small for {count} DE restarts; "
            f"need at least {minimum * count}")
    quotient, remainder = divmod(total, count)
    return [quotient + (1 if i < remainder else 0) for i in range(count)]


def _de_restart_plan(count, seed, de_total, polish_total):
    """Return (restart, seed, DE budget, polish budget) for every requested run."""
    eval_budgets = _split_budget(
        de_total, count, minimum=5, flag="--sweep-de-max-evals")
    polish_budgets = _split_budget(
        max(0, int(polish_total)), count, minimum=0,
        flag="--sweep-polish-max-evals")
    return [
        (restart, int(seed) + restart,
         eval_budgets[restart], polish_budgets[restart])
        for restart in range(int(count))
    ]


def _search_score(rec, gate_field, depth_weight):
    """The optimizer/ranking score, including the behind-camera hard floor."""
    if rec.get("behind_camera"):
        return NEG
    return metrics.score_of(rec, gate_field, depth_weight)


def _search_and_report(p, rast):
    """Score the grid (plus any refine passes), re-render the winner, write the
    report. Everything here runs AFTER `_prepare` resolved the search."""
    ctx, cfg, sess = p.ctx, p.cfg, p.sess
    V = p.V
    objs, canonical, joint_defs = p.objs, p.canonical, p.joint_defs
    explicit_candidates = p.explicit_candidates
    space, sweep_space, ranges = p.space, p.sweep_space, p.ranges
    dof_names, quality = p.dof_names, p.quality
    gate_field, grid_iou = p.gate_field, p.grid_iou
    depth_sup, depth_weight = p.depth_sup, p.depth_weight
    deadline, sl, sw, sh = p.deadline, p.sl, p.sw, p.sh
    strategy = str(cfg.strategy).lower()
    topk_n = _resolved_topk(cfg.topk, strategy, bool(explicit_candidates))

    # ---- the scoring closures ------------------------------------------- #
    scored = []
    bbox_corners = _canonical_corners(canonical, objs)

    def score(cand):
        """Realize -> pose -> render -> IoU (+depth). Appends the rec, returns it."""
        placement = cand["placement"]
        M = _placement_matrix(placement)
        rec = {"placement": placement, "order": cand.get("order"),
               "joints": cand.get("joints"), "iou_raw": 0.0, "iou_visible": None,
               "candidate_id": cand.get("candidate_id"),
               "hypothesis_ids": cand.get("hypothesis_ids") or [],
               "depth_mae": None, "depth_canon": None, "behind_camera": False}
        if _behind_camera(ctx.cam_obj, M, bbox_corners):
            rec["behind_camera"] = True
            scored.append(rec)
            return rec
        # `where=`: pose_frame REFUSES an incomplete joint-state dict, and without a
        # site the message just says "pose_frame" — indistinguishable across the
        # scoring loop, the winner re-render and the panel loop, which is exactly what
        # a reader needs to know first. The candidate id names WHICH candidate.
        scene.pose_frame(objs, canonical, placement, placement["scale"],
                         joint_defs, cand.get("joints") or {},
                         where=f"{V}: scoring candidate "
                               f"{cand.get('candidate_id') or '?'}")
        (rec["iou_raw"], rec["iou_visible"],
         rec["depth_mae"], rec["depth_canon"]) = sess.score_posed(grid_iou,
                                                                  depth_sup)
        scored.append(rec)
        return rec

    visuals = visual_budget.visuals_level(cfg.visuals)
    waived = bool(cfg.waive_visual_budget)
    # This estimate drives progress and the pre-scoring visuals=all budget check.
    # DE's optimizer may converge before its cap, but it cannot promise a lattice
    # count in advance, so use the requested evaluation budgets as the upper bound.
    refine_n = (0 if (explicit_candidates or sl is not None or strategy == "de")
                else max(0, int(cfg.refine)))
    if explicit_candidates:
        search_est = len(explicit_candidates)
        est = search_est
    elif strategy == "de":
        search_est = max(1, int(cfg.de_max_evals))
        est = search_est + (max(0, int(cfg.polish_max_evals))
                            if cfg.polish == "powell" else 0)
    else:
        search_est = planner.grid_budget_estimate(space, ranges)
        est = search_est * (1 + refine_n)
    if visuals == "all":
        # fail BEFORE scoring: a 400-cell grid takes minutes to score, and refusing
        # it afterwards wastes all of them.
        visual_budget.check_budget(
            (topk_n if strategy == "de" and not explicit_candidates else est),
            waived=waived, what=f"{V} visuals=all",
            detail=(f"{topk_n} restart winners from de"
                    if strategy == "de" and not explicit_candidates else
                    f"up to ~{search_est} candidates from {strategy}"
                    + (f" x {1 + refine_n} passes" if refine_n else "")))
    prog = _Progress(est, V, deadline=deadline, user_timeout=cfg.timeout)

    # ---- run the selected search policy --------------------------------- #
    print(f"[render_wrapper] {V}: space={sweep_space} "
          f"dofs={','.join(dof_names)} at {sw}x{sh} (gate={gate_field}, "
          f"quality={quality:g})")
    # Every candidate is STAMPED with the pass that produced it, and every pass
    # records the lattice it came from (see `passes` in the manifest). Without that,
    # a refined winner is unreconstructable from disk: the report kept only the
    # COARSE ranges, so a reader could tell a value was off-lattice but could not
    # draw the grid it actually came from, and `n_candidates` silently read as
    # (1 + refine) x the grid product with nothing to explain the multiple.
    cur_pass = [0]

    def score_pass(items, realize=True):
        """Score one pass's worth of candidates, stamping each with `cur_pass`.

        `realize=True` takes DOF VECTORS off a lattice (`space.realize` maps them to
        placements); `realize=False` takes ready placements (an explicit candidate
        list, which has no lattice). That one call is the only difference between the
        grid path and the candidate path — everything after it (pass stamp, rank,
        progress, behind-camera floor) is identical, and duplicating it meant a
        change to how a candidate is ranked had to land in two loops."""
        for item in items:
            if prog.expired():
                break
            rec = score(space.realize(item) if realize else item)
            rec["_pass"] = cur_pass[0]
            val = _search_score(rec, gate_field, depth_weight)
            prog.update(prog.best if val <= NEG / 2 else max(prog.best, val))

    optimizer = None
    de_winners = None
    if explicit_candidates:
        score_pass(explicit_candidates, realize=False)
        passes = [metrics.pass_record(
            0, {}, len(scored), kind="candidates")]
    elif strategy == "de":
        if int(cfg.de_max_evals) <= 0:
            raise ValueError("--sweep-de-max-evals must be positive")
        if int(cfg.de_popsize) <= 0:
            raise ValueError("--sweep-de-popsize must be positive")
        if not 0.0 <= float(cfg.de_recombination) <= 1.0:
            raise ValueError("--sweep-de-recombination must be in [0,1]")
        if cfg.polish not in ("none", "powell"):
            raise ValueError("--sweep-polish must be none or powell")
        mutation = search_policy.parse_mutation(cfg.de_mutation)
        restart_plan = _de_restart_plan(
            topk_n, cfg.de_seed, cfg.de_max_evals, cfg.polish_max_evals)
        passes = []
        runs = []
        de_winners = []
        for restart, seed, eval_budget, polish_budget in restart_plan:
            if prog.expired():
                break
            before = len(scored)
            pass_offset = len(passes)

            def evaluate_de(x, stage_idx, restart=restart, seed=seed,
                            pass_offset=pass_offset):
                rec = score(space.realize(x))
                rec["_pass"] = pass_offset + int(stage_idx)
                rec["_restart"] = restart
                rec["_seed"] = seed
                value = _search_score(rec, gate_field, depth_weight)
                prog.update(prog.best if value <= NEG / 2
                            else max(prog.best, value))
                return value

            outcome = search_policy.run_de(
                space, ranges, evaluate_de, prog.expired,
                max_evals=eval_budget, popsize=cfg.de_popsize,
                seed=seed, mutation=mutation,
                recombination=cfg.de_recombination, polish=cfg.polish,
                polish_max_evals=polish_budget)
            restart_scored = scored[before:]
            if restart_scored:
                de_winners.append(max(
                    restart_scored,
                    key=lambda rec: _search_score(
                        rec, gate_field, depth_weight)))
            for local_index, stage in enumerate(outcome.stages):
                search_pass = metrics.pass_record(
                    pass_offset + local_index, stage.bounds, stage.n,
                    kind=stage.kind)
                search_pass.update({
                    "restart": restart,
                    "seed": seed,
                    "budget": (eval_budget if stage.kind == "de"
                               else polish_budget),
                })
                passes.append(search_pass)
            runs.append({
                "restart": restart,
                "seed": seed,
                "de_budget": eval_budget,
                "polish_budget": polish_budget,
                **outcome.metadata,
            })
        optimizer = {
            "strategy": "best1bin",
            "seed": int(cfg.de_seed),
            "population_multiplier": int(cfg.de_popsize),
            "mutation": ([float(mutation[0]), float(mutation[1])]
                         if isinstance(mutation, tuple) else float(mutation)),
            "recombination": float(cfg.de_recombination),
            "polish": cfg.polish,
            "restarts_requested": topk_n,
            "restarts_completed": len(de_winners),
            "max_evals_total": int(cfg.de_max_evals),
            "polish_max_evals_total": int(cfg.polish_max_evals),
            "runs": runs,
        }
    else:
        vectors = planner.grid(space, ranges)
        if sl is not None:                   # parallel shard: no refine
            idx, tot = sl
            vectors = vectors[idx::tot]
            print(f"[render_wrapper] {V}: grid slice {idx}/{tot} -> "
                  f"{len(vectors)} candidates (refine disabled)")
        score_pass(vectors)
        passes = [metrics.pass_record(
            0, ranges, len(scored), kind="coarse",
            slice_of=(list(sl) if sl is not None else None))]
    # coarse->fine refinement: re-center on the best VECTOR, shrink, re-score.
    cur = ranges
    if refine_n:
        # `pass_idx`, not `p`: `p` is this function's _Prep, still needed by
        # _write_the_report / _render_winner_and_panels below, so the loop counter
        # must not shadow it.
        for pass_idx in range(refine_n):
            if prog.expired():
                break
            best_rec = max(
                scored,
                key=lambda rec: _search_score(rec, gate_field, depth_weight))
            best_x = _best_vector(space, best_rec)
            cur = planner.refine_ranges(space, cur, best_x,
                                         float(cfg.refine_shrink))
            grid = planner.grid(space, cur)
            print(f"[render_wrapper] {V}: refine pass {pass_idx + 1}/{cfg.refine} "
                  f"{len(grid)} candidates around best "
                  f"{metrics.score_of(best_rec, gate_field, depth_weight):.4f}")
            before = len(scored)
            cur_pass[0] = pass_idx + 1
            score_pass(grid)
            # the CENTER is what makes the window reconstructable: `cur` is the band,
            # and the vector it was re-centered on says which candidate it descended
            # from. `n` is what was actually scored, so a timeout mid-pass is visible
            # rather than implied by a short candidate list.
            passes.append(metrics.pass_record(
                pass_idx + 1, cur, len(scored) - before, kind="refine",
                center=metrics.pass_center(space, best_x),
                shrink=float(cfg.refine_shrink)))

    prog.done()
    if depth_sup is not None:
        depth_sup.teardown()
    sess.end_scoring()
    timed_out = prog.expired()

    ranked = sorted(
        de_winners if de_winners is not None else scored,
        key=lambda r: _search_score(r, gate_field, depth_weight),
        reverse=True)
    for rank, rec in enumerate(ranked):
        rec["_rank"] = rank

    _write_the_report(p, ranked, scored, passes, ranges, timed_out,
                      optimizer=optimizer, topk_n=topk_n)


def _dof_stats_for_report(strategy, explicit_candidates, scored, swept_kinds,
                          gate_field, depth_weight):
    """Return per-DOF projections where the sampling design supports them.

    DE is adaptive and continuous: nearly every evaluation has a distinct value on
    every axis, so this projection is candidate scatter rather than a landscape.
    """
    if str(strategy).lower() == "de" and not explicit_candidates:
        return None
    return metrics.dof_landscape_stats(
        scored, swept_kinds, gate_field, depth_weight)


def _write_the_report(p, ranked, scored, passes, ranges, timed_out,
                      optimizer=None, topk_n=None):
    """Assemble the manifest and write <stem>.json / .txt (via `metrics`).

    Split from the search because it is a different KIND of work: the search decides
    what is true, this decides what to say about it. It also owns the empty case —
    a timed-out or degenerate grid still writes a minimal report rather than nothing,
    which is what makes a failed sweep debuggable from disk.

    `ranges` and `passes` are passed rather than read off `p`: they are what the
    SEARCH produced (refinement appends passes and re-centres the band), not what
    preparation resolved, and reading a stale `p.ranges` here would describe a
    lattice the winner never sat on."""
    ctx, cfg, sess = p.ctx, p.cfg, p.sess
    a, V, STEM = p.a, p.V, p.STEM
    all_joint_names, base_states = p.all_joint_names, p.base_states
    start_placement, explicit_candidates = p.start_placement, p.explicit_candidates
    sweep_space = p.sweep_space
    dof_names, kinds, quality = p.dof_names, p.kinds, p.quality
    gate_field, depth_weight = p.gate_field, p.depth_weight
    sw, sh = p.sw, p.sh
    visuals = visual_budget.visuals_level(cfg.visuals)
    waived = bool(cfg.waive_visual_budget)
    swept_kinds = {d: kinds[d] for d in dof_names}
    topk_n = int(topk_n if topk_n is not None else
                 _resolved_topk(cfg.topk, str(cfg.strategy).lower(),
                                bool(explicit_candidates)))
    # HELD = every joint the grid did not actually vary: the ones outside the space
    # AND the ones in the space but in no range (the grid pins those at their base
    # state — see planner.grid's kind-dependent hold). Reporting the second group
    # as "swept" would claim a search that never happened; explicit candidates carry
    # absolute joints per candidate, so nothing is pinned there.
    ranged = set(dof_names) if explicit_candidates else set(ranges)
    held_joints = {n: round(base_states[n], 6)
                   for n in all_joint_names if n not in ranged}
    meta_common = {
        "view": V,
        "frame": ctx.frame.name,
        "view_index": getattr(ctx.frame, "view_index", None),
        "camera_c2w": (np.asarray(ctx.frame.camera_c2w).tolist()
                       if getattr(ctx.frame, "camera_c2w", None) is not None else None),
        "intrinsics_provenance": ctx.frame_intr_provenance,
        "space": sweep_space,
        "method": ("candidates" if explicit_candidates
                   else str(cfg.strategy).lower()),
        # the raw --sweep-start-shift order, as asked for (`pose_start` in the
        # report already reflects the SHIFTED placement).
        "start_shift": cfg.start_shift or None,
        # WHERE the roll/yaw/pitch increments orbited. A recorded order ("yaw: 30")
        # is uninterpretable without it — an angle means nothing without the point
        # it turned about — and serializing `source` is what makes a future change
        # to the pivot's DEFINITION detectable instead of silently re-interpreting
        # every archived order. None for a joint-only space (no order geometry).
        "pivot": _pivot_out(p.pivot),
        # refine passes only (pass 0 is the coarse grid), derived from `passes` rather
        # than counted separately — one number, one source.
        "n_passes": sum(1 for search_pass in passes
                        if search_pass.get("kind") == "refine"),
        "swept_dofs": dof_names,
        "held_joints": held_joints,
        # The originally requested search domain. Grid passes add their exact
        # lattices; DE/polish stages retain these continuous bounds.
        "ranges": {k: list(v) for k, v in ranges.items()},
        "passes": passes,
        "sweep_resolution": [sw, sh],
        "match_resolution": [a.base_w, a.base_h],
        "quality": quality,
        "mask": cfg.mask,
        "hand_mask": cfg.hand_mask or None,
        "gate_field": gate_field,
        "depth_weight": depth_weight or None,
        "timed_out": timed_out,
        "n_candidates": len(scored),
        "topk": topk_n,
        **({"optimizer": optimizer} if optimizer is not None else {}),
    }

    if not ranked:
        sess.restore()
        meta = {**meta_common,
                "status": "timed_out" if timed_out else "no_candidates"}
        metrics.write_empty_report(a.out, start_placement, meta, out_stem=STEM)
        print(f"[render_wrapper] {V}: no candidates scored "
              f"({'timed out' if timed_out else 'empty grid'}); wrote minimal "
              f"{STEM}.json")
        return
    best = ranked[0]
    # For DE, top-K is both the restart count and the panel set: one winner from
    # every completed restart. Grid/candidate listings remain independent of their
    # panel plan, which selects by ordered cell or explicit score order.
    budget_error = None
    try:
        if str(cfg.strategy).lower() == "de" and not explicit_candidates:
            retained = [] if visuals == "none" else list(ranked)
            dump_n = len(retained)
            selection = metrics.BY_RESTART
            cells = None
            if retained:
                visual_budget.check_budget(
                    len(retained), waived=waived,
                    what=f"{V} DE restart winners",
                    detail=f"{len(retained)} top-K restart winners")
        else:
            retained, dump_n, selection, cells = metrics.panel_plan(
                ranked, swept_kinds, ranges, cfg.dump_topk, visuals=visuals,
                waived=waived, what=V)
    except visual_budget.VisualBudgetError as exc:
        # The sweep is already SCORED. A panel-count refusal must not throw that
        # away: fall back to the auto level (winner + best-per-ordered-cell) and
        # record the refusal in the report, so the agent sees why it got fewer
        # images and can re-run with the waiver if it meant it.
        budget_error = str(exc)
        print(f"[render_wrapper] {V}: panel set refused ({budget_error}); "
              "falling back to visuals=auto — the scores are kept")
        visuals = "auto"
        if str(cfg.strategy).lower() == "de" and not explicit_candidates:
            retained, dump_n = list(ranked[:1]), min(1, len(ranked))
            selection, cells = metrics.BY_RESTART, None
        else:
            retained, dump_n, selection, cells = metrics.panel_plan(
                ranked, swept_kinds, ranges, cfg.dump_topk, visuals="auto",
                waived=waived, what=V)

    dump_images = _render_winner_and_panels(
        p, best, retained, dump_n, selection, capture_depth=visuals != "none")

    # A continuous DE trace does not support a per-DOF landscape: almost every
    # sample has a unique value on every axis, so the would-be marginal is just
    # candidate scatter and its "boundary" is only the sampled min/max. Keep these
    # statistics for lattice/candidate reports, but do not publish them to DE agents.
    dof_stats = _dof_stats_for_report(
        cfg.strategy, explicit_candidates, scored, swept_kinds,
        gate_field, depth_weight)
    lattice_dofs = [d for d in dof_names
                    if d in ranges and len(ranges[d]) >= 3
                    and int(ranges[d][2]) >= 2]
    # ONE key for the dense evidence, keyed "yaw|pitch" for a pair and "yaw" for the
    # single-DOF marginal, both in the same shape.
    # The candidates are already in memory and each field is one pass over them.
    score_fields, fields_omitted = metrics.score_fields(
        scored, lattice_dofs, kinds, gate_field, depth_weight)

    meta = {
        **meta_common,
        **({"dof_stats": dof_stats} if dof_stats is not None else {}),
        "score_fields": score_fields or None,
        **({"score_fields_omitted": fields_omitted} if fields_omitted else {}),
        # cells_occupied is the distinct-cell count over ALL scored candidates, not
        # the number panelled, so a short sheet is attributable: fewer cells than
        # asked means a degenerate grid, cells > rendered means dump_topk capped it.
        # None when the order has no lattice (an explicit candidate set).
        "panels": {"asked": dump_n, "rendered": len(dump_images),
                   "visuals": visuals, "selection": selection,
                   "cells_occupied": cells,
                   **({"budget_error": budget_error} if budget_error else {})},
        "status": "timed_out" if timed_out else "ok",
    }
    # Grid/candidates pass the full score order. DE passes the score-ordered winner
    # from each completed restart. Panelled records carry _panel_image.
    metrics.write_report(a.out, ranked, start_placement, best, meta, topk_n,
                          gate_field, depth_weight, swept_kinds, dump_images,
                          out_stem=STEM, view=V)


def _render_winner_and_panels(
        p, best, retained, dump_n, selection, capture_depth=True):
    """Re-render the winner (and any panels) at FULL match res; -> panel filenames.

    Scoring ran at the low `--sweep-quality` resolution, so every image a human
    looks at is produced here, after the search is over. Split out of `run` because
    it is the one phase that is purely side-effecting — it renders and returns
    filenames, reads nothing back into the ranking — so keeping it inline invited
    reading the scoring loop and the presentation loop as one thing. `panel_plan`
    already decided WHAT to render; this only draws it, off the same `_Prep` every
    other phase reads.

    Each `pose_frame` gets a distinct `where=`: the message is otherwise identical
    across the scoring loop, the winner re-render and the panel loop, and which one
    refused an incomplete joint dict is the first thing a reader needs."""
    ctx, sess, a, V, STEM = p.ctx, p.sess, p.a, p.V, p.STEM
    objs, canonical, joint_defs = p.objs, p.canonical, p.joint_defs
    sess.restore()
    sess.set_full_res()
    render.place_match_camera(a, ctx.cam_obj, ctx.center, ctx.radius, ctx.frame_intr)
    rider = None
    if capture_depth and depth_config.depth_render_enabled(a.out):
        try:
            rider = depth_view.RenderedDepthRider(
                a.out, stem=f"_{STEM}_panel_depth_tmp")
            rider.setup()
        except Exception as exc:
            if rider is not None:
                rider.teardown()
            print(f"[render_wrapper] {V}: candidate depth unavailable "
                  f"({exc}); continuing with RGB panels")
            rider = None

    def capture(record, key, image_path, depth_name):
        nonlocal rider
        if rider is None:
            return
        try:
            rider.save(image_path, os.path.join(a.out, depth_name))
            record[key] = depth_name
        except Exception as exc:
            print(f"[render_wrapper] {V}: candidate depth unavailable "
                  f"({exc}); continuing with RGB panels")
            rider.teardown()
            rider = None

    dump_images = []
    try:
        scene.pose_frame(
            objs, canonical, best["placement"], best["placement"]["scale"],
            joint_defs, best.get("joints") or {},
            where=f"{V}: re-rendering the WINNER "
                  f"{best.get('candidate_id') or '?'} at full match res")
        best_image = os.path.join(a.out, f"{STEM}_best.png")
        render.render_to(best_image)
        capture(best, "_best_depth", best_image, f"{STEM}_best_depth.npy")

        if dump_n:
            short = ("" if len(retained) >= dump_n else
                     f" (asked {dump_n}; only {len(retained)} distinct ordered "
                     "cells are occupied)")
            print(f"[render_wrapper] {V}: dumping {len(retained)} "
                  f"{selection} candidates to {STEM}_top_*.png{short}")
            # `retained` is already capped at dump_n by panel_plan.
            for i, r in enumerate(retained):
                scene.pose_frame(
                    objs, canonical, r["placement"], r["placement"]["scale"],
                    joint_defs, r.get("joints") or {},
                    where=f"{V}: panelling candidate "
                          f"{r.get('candidate_id') or '?'} "
                          f"({STEM}_top_{i:02d}.png)")
                fname = f"{STEM}_top_{i:02d}.png"
                image_path = os.path.join(a.out, fname)
                render.render_to(image_path)
                dump_images.append(fname)
                # tag the RECORD, not the index: the listing is now the full score
                # order, so a panel's row is wherever that candidate ranks.
                r["_panel_image"] = fname
                capture(r, "_panel_depth", image_path,
                        f"{STEM}_top_{i:02d}_depth.npy")
    finally:
        if rider is not None:
            rider.teardown()

    sess.restore()
    return dump_images


def _parse_slice(spec):
    """'i/K' -> (i, K) with 0 <= i < K; None if empty/invalid (a parallel shard)."""
    if not spec or not spec.strip():
        return None
    try:
        i, k = (int(x) for x in spec.split("/"))
    except ValueError:
        raise ValueError(f"--sweep-grid-slice: want 'INDEX/TOTAL', got {spec!r}")
    if not (0 <= i < k):
        raise ValueError(f"--sweep-grid-slice: need 0 <= INDEX < TOTAL, got {spec!r}")
    return i, k


def _best_vector(space, rec):
    """Recover the DOF vector `x` (aligned to space.dof_names) from a scored rec —
    pose orders from rec['order'], joint states from rec['joints'] — so a grid
    refine pass can re-center on the best in the fixed vector space.

    The MISSING-key fallback is kind-dependent for the same reason as
    `planner.grid`'s held fill: a pose order is an increment (0 = unmoved) but a
    joint state is ABSOLUTE, so a joint absent from the rec falls back to its base
    state — falling back to 0 would re-center the refine window on a straightened
    joint the scored render never showed.

    There is no THIRD fallback behind the base state: `space.base_states` covers
    every joint in the space by construction (build_space is handed the base fill
    for all of them), so a joint missing from BOTH the rec and the base is a bug
    in the space, not an input to guess at. It raises, because the alternative is
    re-centering the refine window on a configuration nothing ever rendered."""
    order = rec.get("order") or {}
    joints = rec.get("joints") or {}
    x = []
    for d in space.dof_names:
        if space.kinds[d] == dof_space.POSE:
            x.append(float(order.get(d, 0.0)))   # increment: 0 IS unmoved
        elif d in joints:
            x.append(float(joints[d]))
        elif d in space.base_states:
            x.append(float(space.base_states[d]))
        else:
            raise ValueError(
                f"refine: joint {d!r} has no state in the scored candidate and "
                "none in the space's base states, so the window cannot be "
                "re-centered. Joint states are absolute — there is no value "
                "that means 'unchanged'.")
    return np.array(x, dtype=float)


def _resolve_pose_start(cfg, ctx):
    """--sweep-pose-start JSON placement if given, else the posed frame's placement.

    WHERE THE GRID STARTS: the placement its offsets are measured from, i.e. the
    candidate at offset (0,0,0).

    Its SCALE always comes from the FRAME, never from the placement itself: scale
    is the object's shared SCALE and no verb may search it (conventions/POSE.md §3),
    so the only scale a sweep may score at is the one the scene declares. A
    `--sweep-pose-start` that omits scale is the common case — a fragment entry, a
    progress.json pose and a paste block all omit it by design
    (conventions/state_json.md §4).

    A pose start that states a CONFLICTING scale is refused rather than honoured or
    silently overridden: it means the caller believes scale is theirs to set, and
    both of the other answers leave that belief in place."""
    frame_scale = (ctx.placement or {}).get("scale") if ctx.placement else None
    if not cfg.pose_start:
        return ctx.placement
    import json
    if os.path.isfile(cfg.pose_start):
        with open(cfg.pose_start) as f:
            start = json.load(f)
    else:
        start = json.loads(cfg.pose_start)
    if not isinstance(start, dict):
        raise ValueError(f"--sweep-pose-start must be a placement object, got "
                         f"{type(start).__name__}")
    start = dict(start)
    stated = start.get("scale")
    if frame_scale is None:
        # no posed frame to take the shared SCALE from: the placement must carry it.
        if stated is None:
            raise ValueError(
                "--sweep-pose-start carries no 'scale' and there is no posed frame to "
                "take the shared SCALE from. State 'scale' in it.")
        return start
    if stated is not None and abs(float(stated) - float(frame_scale)) > 1e-9:
        raise ValueError(
            f"--sweep-pose-start states scale {float(stated)} but the frame's shared "
            f"SCALE is {float(frame_scale)}. Scale belongs to the OBJECT, not to a "
            "placement, and no sweep verb may search it — resize via the top-level "
            "SCALE in scene.py. Drop 'scale' from the pose start (it is filled from "
            "the frame) or correct it.")
    start["scale"] = frame_scale
    return start


def _apply_start_shift(cfg, start_placement, base_states, joint_defs, canonical, objs,
                       place, sw, sh):
    """Compose the --sweep-start-shift directional order onto the pose start ->
    a SHIFTED start.

    Reuses the `apply` grammar (`planner.parse_apply_pairs`) and the `realize` seam:
    it builds a one-off Space from the ORIGINAL start (so the shift is measured from
    where the frame began), realizes the shift vector, and returns the shifted
    (placement, joint states, scale) for the sweep to search AROUND. Carries joints,
    unlike --sweep-pose-start. Returns (start_placement, base_states, base_scale_val)."""
    pairs = planner.parse_apply_pairs(cfg.start_shift)
    defs = {j.get("name"): j for j in joint_defs if j.get("name")}
    shift_pose = [d for d in pairs if dof_space.classify_dof(d) == dof_space.POSE]
    shift_joint_names = [d for d in pairs if dof_space.classify_dof(d) == dof_space.JOINT]
    unknown = [n for n in shift_joint_names if n not in defs]
    if unknown:
        raise ValueError("--sweep-start-shift: unknown joint(s) " + ", ".join(unknown) +
                         f" (declared: {', '.join(defs) or 'none'})"
                         + "".join(planner.canon_cross_hint(n) for n in unknown))
    shift_joint_defs = [defs[n] for n in shift_joint_names]
    # the shifting hop's own pivot is discarded: it orbits the ORIGINAL start (which
    # is what makes the hop measured from where the frame started), while the search
    # that follows rebuilds P0/OrderCtx on the SHIFTED start and records THAT pivot —
    # the one every scored candidate actually turned about.
    shift_space, _hop_pivot = build_space(
        shift_pose, shift_joint_defs, start_placement, base_states, canonical,
        objs, place, sw, sh, joint_defs=joint_defs)
    x = np.array([float(pairs[d]) for d in shift_space.dof_names], dtype=float)
    cand = shift_space.realize(x)
    shifted = transforms.normalize_placement(cand["placement"])
    return shifted, cand["joints"], shifted["scale"]


def _realize_explicit_candidates(spec, start_placement, base_states, base_scale,
                                 joint_defs):
    """Turn parsed absolute candidates into the engine's internal record form."""
    items = planner.parse_candidates(spec)
    if not items:
        return []
    defs = {j.get("name"): j for j in joint_defs if j.get("name")}
    out = []
    for item in items:
        placement = transforms.normalize_placement(item.get("pose") or start_placement)
        placement["scale"] = base_scale
        states = dict(base_states)
        supplied = item.get("joints") or {}
        unknown = sorted(set(supplied) - set(defs))
        if unknown:
            raise ValueError("candidate declares unknown joint(s): " +
                             ", ".join(unknown))
        for name, value in supplied.items():
            value = float(value)
            limit = transforms.joint_limit(defs[name])
            if limit is not None:
                value = min(max(value, limit[0]), limit[1])
            states[name] = value
        out.append({
            "placement": placement,
            "order": None,
            "joints": states,
            "candidate_id": item["candidate_id"],
            "hypothesis_ids": item.get("hypothesis_ids") or [],
        })
    return out
