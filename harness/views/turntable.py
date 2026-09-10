"""turntable view — orbiting camera views per ARTICULATION STATE.

A genuine camera orbit: a 3D coherence check of the shape from other angles,
independent of the match pose. Because the
object articulates, coherence must hold in EVERY observed state — a part can read
connected when a door is closed but detached when it opens — so we render the
whole view set once per UNIQUE joint state across the frames:

    turntable_<state>_az<AAA>_el<+EE>.png   (state = a label per unique joint config)

WHICH DIRECTIONS, and what the files are called, live in `core.turntable_views` —
shared with `analysis.viz.turntable_sheet`, which lays these renders out on a page
and must agree about both. The default set (`sphere8`) is a level 4-azimuth ring
plus a raised pair and a dropped pair: a level ring alone is blind to a caved-in
top or an unbuilt base.

State posing uses the reference frame's base pose (so the object reads naturally
upright, as posed) with each state's joint states applied on top; the base pose
is a rigid transform that does not affect connectivity, so this stays a pure
shape check. With no joints there is a single 'rest' state.
Restores the base resolution + FOV (the match view may have changed these via
per-frame intrinsics) and the parts' world matrices afterwards, so
match/depth/pose.json/GLB are unaffected.
"""

import math
import os

import bpy

from core import joints as joints_core
from core import turntable_views
from rig import camera, render, scene


def _unique_states(spec):
    """Ordered list of (label, joint_states) — one per distinct configuration
    across all frames. Falls back to a single 'rest' state when there are no
    joints / no per-frame states.

    Each emitted dict names EVERY declared joint, which is what makes it a legal
    pose_frame argument (states are absolute, so a partial dict is refused, not
    defaulted — core.joints.require_complete_states).

    The 0.0 fill is SPLIT by kind rather than applied to every name behind a comment
    claiming it is safe, because a comment is not a constraint. Articulated joints go
    through require_complete_states and raise naming the frame; a `fixed` joint is
    identity at any state, so its 0.0 is arithmetic, not a guess."""
    if not spec.joint_defs:
        return [("rest", {})]
    names = [j.get("name") for j in spec.joint_defs if j.get("name")]
    articulated = joints_core.articulated_names(spec.joint_defs)
    fixed = [n for n in names if n not in set(articulated)]
    seen = {}
    order = []
    for fname, entry in spec.frames.items():
        st = joints_core.require_complete_states(
            spec.joint_defs, entry.get("joints", {}) or {},
            f"turntable: frame {fname!r}")
        state = {n: float(st[n]) for n in articulated}
        state.update({n: float(st.get(n, 0.0)) for n in fixed})  # identity anyway
        # signature over the declared joints (rounded so tiny diffs don't split)
        sig = tuple(round(state[n], 4) for n in names)
        if sig not in seen:
            seen[sig] = state
            # label by the first frame that exhibits this state
            order.append((os.path.splitext(str(fname))[0], state))
    # joints but no frames: nothing declared a state, so nothing to carry. Rest is an
    # AUTHORED choice here, spelled out rather than handed over as an empty dict.
    return order or [("rest", {n: 0.0 for n in names})]


def resolve_views(a):
    """The (azimuth, elevation) set this pass renders, from the --turntable-* flags.

    `--elevation` names the level RING and `--azimuth` rotates the whole set (which
    is what that flag's help has always claimed and, until now, never did: the four
    angles were literals here and nothing read it)."""
    return turntable_views.view_set(
        getattr(a, "turntable_views", turntable_views.DEFAULT_VIEW_SET),
        ring_el=a.elevation, az_offset=a.azimuth,
        jitter=getattr(a, "turntable_jitter", None))


def render_view(ctx):
    a = ctx.args
    bpy.context.scene.render.resolution_x = a.base_w
    bpy.context.scene.render.resolution_y = a.base_h
    bpy.context.scene.render.pixel_aspect_x = 1.0
    bpy.context.scene.render.pixel_aspect_y = 1.0
    ctx.cam_obj.data.type = "ORTHO" if a.ortho else "PERSP"
    ctx.cam_obj.data.lens_unit = "FOV"
    ctx.cam_obj.data.angle = math.radians(a.fov)
    ctx.cam_obj.data.shift_x = 0.0
    ctx.cam_obj.data.shift_y = 0.0
    views = resolve_views(a)

    objs = scene.mesh_parts()
    canonical = ctx.canonical or {o: o.matrix_world.copy() for o in objs}
    saved = {o: o.matrix_world.copy() for o in objs}
    states = _unique_states(ctx.spec) if ctx.spec is not None else [("rest", {})]
    # orient the object as posed in the current frame (its rotation reads
    # naturally upright); the per-state joints are applied on top. The base pose
    # is rigid, so this does not affect the connectivity we are checking.
    base_pose = ctx.frame.pose if ctx.frame is not None else None
    scale = ctx.frame.scale if ctx.frame is not None else 1.0
    joint_defs = ctx.spec.joint_defs if ctx.spec is not None else []

    print(f"[render_wrapper] turntable: {len(states)} state(s) x {len(views)} "
          f"view(s) [{turntable_views.describe(views)}]")

    try:
        for label, joint_states in states:
            scene.pose_frame(objs, canonical, base_pose, scale, joint_defs,
                             joint_states, where=f"turntable state {label!r}")
            center, radius = camera.scene_bbox()
            # names built for the WHOLE state up front: image_names refuses a set
            # whose views round to the same filename, and finding that out before
            # rendering beats finding out by one render overwriting another.
            names = turntable_views.image_names(label, views)
            for (az, el), name in zip(views, names):
                camera.place_camera(ctx.cam_obj, center, radius, az, el,
                                    a.radius, a.ortho)
                render.render_to(os.path.join(a.out, name))
    finally:
        scene.restore_world(objs, saved)
