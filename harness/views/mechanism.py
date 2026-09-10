"""mechanism view — one joint swept across its LIMIT twice, as a blind A/B pair.

The turntable orbits the camera and holds articulation; this holds the camera and
sweeps ONE joint, so a wrong `axis` sign shows as an arc going the wrong way (why
that needs its own view: views/mechanism.md).

Each joint is swept TWICE — once about the declared `axis`, once about its
negation — onto blind `A`/`B` labels, because "which of these two is the object
opening" is a forced choice with a right answer while "does this arc look right"
is a question that answered yes over a mirrored hinge.

    mechanism_<joint>_<arm>_r<R>_s<NN>_<state>.png

Rendered in the CANONICAL frame (no base pose) — the frame the declaration under
inspection is written in. Every other joint is held at rest (0.0 clamped into
its own limit; states are absolute), so one strip is one joint's doing.
Viewpoints are picked per joint by `core.mechanism_views.pick_views`; each row
keeps its own fixed camera, framed once to the full swept extent so the group
cannot walk out of shot or change scale between tiles.

Restores the parts' world matrices afterwards, so match/depth/pose.json/GLB are
unaffected — same contract as the turntable.
"""

import math
import os

import bpy

from core import joints as joints_core
from core import mechanism_calls
from core import mechanism_views
from rig import camera, render, scene, transforms




def _articulated(spec):
    """The joints this view sweeps: everything with a DOF, in declaration order.

    `fixed` joints are excluded — identity at any state, so their "arc" is nine
    identical renders."""
    if spec is None or not spec.joint_defs:
        return []
    names = set(joints_core.articulated_names(spec.joint_defs))
    return [j for j in spec.joint_defs if j.get("name") in names]


def _held_states(joint_defs, active):
    """Every joint but `active` pinned at rest, clamped into its own limit.

    Clamped rather than set to a bare 0.0 because a joint's declared range need
    not contain zero (a limit of (10, 80) is legal), and `pose_frame` would refuse
    — correctly — to render a state outside it. Includes `fixed` joints so the
    dict names every declared joint, which is what makes it a legal argument
    (states are absolute; a partial dict is refused, not defaulted)."""
    return {j.get("name"): transforms.clamp_joint_state(j, 0.0)
            for j in joint_defs if j.get("name") != active}


def _arm_defs(joint_defs, active, axis):
    """`joint_defs` with `active`'s axis replaced — how the mirrored arm is
    swept. A copy, because the spec's own defs are what every other view and
    pose.json are about; only this view's two arcs differ."""
    out = []
    for j in joint_defs:
        if j.get("name") == active:
            j = dict(j)
            j["axis"] = tuple(axis)
        out.append(j)
    return out


def _swept_bounds(objs, canonical, arm_defs, active, states):
    """(center, radius) framing the union of the object over EVERY arm's sweep.

    Framed ONCE across both arms. Per-state framing would zoom between tiles, so
    the arc would partly be the camera's; per-ARM framing is worse — the arms
    sweep different volumes, so one strip would render larger and the reader could
    tell them apart WITHOUT LOOKING AT THE MECHANISM.

    Unions AABBs and spheres once at the end: boxing a sphere and re-sphering
    inflates the radius by up to sqrt(3) per round trip (that shrank the object to
    ~40% of frame in the first version)."""
    mins = maxs = None
    for defs in arm_defs:
        for state in states:
            joint_states = dict(_held_states(defs, active))
            joint_states[active] = state
            scene.pose_frame(objs, canonical, None, 1.0, defs, joint_states,
                             where=f"mechanism {active!r} framing at {state:+.1f}")
            box = camera.scene_aabb()
            if box is None:
                continue
            lo, hi = box
            mins = lo if mins is None else type(lo)(map(min, mins, lo))
            maxs = hi if maxs is None else type(hi)(map(max, maxs, hi))
    if mins is None:                      # no mesh at all — let the camera default
        return camera.scene_bbox()
    return camera.bbox_to_sphere(mins, maxs)


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

    joint_defs = ctx.spec.joint_defs if ctx.spec is not None else []
    joints = _articulated(ctx.spec)
    samples = int(getattr(a, "mechanism_samples", None)
                  or mechanism_views.DEFAULT_SAMPLES)
    rows = int(getattr(a, "mechanism_rows", None)
               or mechanism_views.DEFAULT_ROWS)

    if not joints:
        # A rigid object has no mechanism to check. Said out loud rather than
        # writing an empty dir: the sheet builder's "no renders" and "nothing to
        # render" are different facts, and only this side knows which it is.
        print("[render_wrapper] mechanism: no articulated joints — nothing to "
              "sweep (a rigid object's shape gate is the turntable)")
        return

    objs = scene.mesh_parts()
    canonical = ctx.canonical or {o: o.matrix_world.copy() for o in objs}
    saved = {o: o.matrix_world.copy() for o in objs}

    print(f"[render_wrapper] mechanism: {len(joints)} joint(s) x "
          f"{len(mechanism_views.ARMS)} arm(s) x {rows} row(s) x {samples} "
          "state(s), canonical frame, viewpoints picked per joint")

    try:
        for joint in joints:
            name = joint.get("name")
            limit = joints_core.joint_limit(joint, strict=True)
            states = mechanism_views.sample_states(limit, samples)
            axis = joint.get("axis", (0.0, 0.0, 1.0))
            # The two arcs: the declared axis and its negation, on blind labels.
            # WHICH label is declared comes from the declaration hash
            # (core.mechanism_calls.arm_axes) — this side just sweeps what it is
            # handed, and deliberately does NOT print the mapping.
            axes = mechanism_calls.arm_axes(joint)
            views = mechanism_views.pick_views(
                axis, joint.get("type", "revolute"), rows=rows)
            print(f"[render_wrapper] mechanism "
                  f"{mechanism_views.describe(name, states, views)}"
                  f"{'' if limit else ' (no declared limit: a full turn)'}")

            # framed ONCE across BOTH arms — the same (center, radius) serves
            # every arm and row, so the object is the same size in every strip.
            # Per-arm framing would leak which arm is which through scale alone
            # (see _swept_bounds), defeating the blinding without anyone
            # looking at a hinge.
            arm_defs = {arm: _arm_defs(joint_defs, name, arm_axis)
                        for arm, arm_axis in axes.items()}
            center, radius = _swept_bounds(objs, canonical,
                                           list(arm_defs.values()), name, states)
            for arm in mechanism_views.ARMS:
                defs = arm_defs.get(arm)
                if defs is None:
                    continue
                held = _held_states(defs, name)
                for row, (az, el) in enumerate(views):
                    # names built for the WHOLE row up front: image_names refuses
                    # a colliding set, and finding that out before rendering
                    # beats finding out by one render overwriting another.
                    files = mechanism_views.image_names(name, arm, row, states)
                    # off-plane angle is a property of the SWING PLANE, which the
                    # two arms share (negating an axis does not move its plane) —
                    # so one number is right for both arms, and printing it per
                    # arm cannot leak the mapping.
                    off_plane = mechanism_views.swing_plane_angle_deg(
                        az, el, axis)
                    print(f"[render_wrapper] mechanism {name!r} arm {arm} "
                          f"row {row}: az{az:.0f}/el{el:+.0f}, "
                          f"{off_plane:.0f} deg off the swing plane")
                    camera.place_camera(ctx.cam_obj, center, radius, az, el,
                                        a.radius, a.ortho)
                    for state, fname in zip(states, files):
                        joint_states = dict(held)
                        joint_states[name] = state
                        scene.pose_frame(
                            objs, canonical, None, 1.0, defs, joint_states,
                            where=f"mechanism {name!r} arm {arm} at "
                                  f"{state:+.1f}")
                        render.render_to(os.path.join(a.out, fname))
    finally:
        scene.restore_world(objs, saved)
