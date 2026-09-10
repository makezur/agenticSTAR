"""
scene_example.py — a RUNNABLE worked example of the scene.py contract.

This file SHOWS the contract by example; it does not define it. The rules live
(once) in the docs — read them there, not here:
  * conventions/POSE.md      — frames, pose (quaternion + translation), M = T·R·S, scale/units
  * conventions/JOINTS.md    — the JOINTS schema, per-frame joint states, limits, forward kinematics
  * harness/shapes/README.md — the build helpers + geometry rules (watertight parts, booleans)
  * AGENT_TASK.md            — the loop, what the harness owns, shape-vs-pose-vs-joint

The harness execs this file, calls build(), then poses the result per frame
(that frame's `pose` + the shared SCALE) and applies joint forward kinematics.

This example models a simple mug — a red ceramic cup (cylinder body + torus
handle), built canonical (+Z up, centered) and posed upright facing camera 0.
The comments below explain THIS example's specific choices; for the general
convention behind any of them, follow the pointer to the doc.
"""

import bpy

# The build-time helpers are SHARED — import them, don't redefine them here. They
# live in the `shapes` package (harness/shapes/); harness/ is on sys.path when the
# harness exec's this scene, so this import resolves. See harness/shapes/README.md.
from shapes import box, convex_hull, set_color, apply_boolean  # noqa: F401


def build():
    """Build the mug in the CANONICAL frame: +Z up, centered, longest dim ~= 1.

    No camera awareness — the mug just stands upright at the origin. Blender's
    cylinder axis is already local +Z, which IS canonical up, so there is no
    orientation trickery here. The per-frame pose (below) handles facing the camera.

    This mug is round (cylinder + torus), so it doesn't need box() — but for any
    axis-aligned part use the box(name, center, size) helper imported from `shapes`
    (it sizes to FULL extent; do NOT hand-roll `scale = size/2`, which halves the
    part).
    """
    # --- body: outer cylinder with an inner cylinder cut out (a cup) ---------
    # depth 1.0 along +Z (canonical up), centered at the origin.
    bpy.ops.mesh.primitive_cylinder_add(radius=0.45, depth=1.0,
                                         location=(0.0, 0.0, 0.0))
    body = bpy.context.active_object
    body.name = "body"

    # inner cavity: slightly smaller, raised toward the top (+Z) so it leaves a
    # solid base.
    bpy.ops.mesh.primitive_cylinder_add(radius=0.37, depth=0.9,
                                         location=(0.0, 0.0, 0.12))
    cavity = bpy.context.active_object
    apply_boolean(body, cavity, "DIFFERENCE")

    # --- handle: a torus offset to the side (+X) of the body -----------------
    # This handle is rigidly attached, so a small intentional intersection seats
    # it convincingly in the cup wall: its inner edge reaches
    # x = 0.55 - 0.22 = 0.33, inside the body radius 0.45. This is useful for this
    # fixed attachment; articulated parent/child meshes may instead have designed
    # clearance, with JOINTS providing their kinematic attachment.
    bpy.ops.mesh.primitive_torus_add(
        major_radius=0.22, minor_radius=0.055,
        location=(0.55, 0.0, 0.0),
    )
    handle = bpy.context.active_object
    handle.name = "handle"
    # torus lies in its local XY plane; stand it up in the canonical XZ plane so
    # the ring is beside the body (its hole faces canonical -Y).
    handle.rotation_euler = (1.5708, 0.0, 0.0)

    # --- convex_hull demo (commented so it doesn't change the mug render) -----
    # For a convex shape the fixed primitives can't express, hull a point set.
    # CONVEX ONLY — carve any concavity with a DIFFERENCE boolean afterward;
    # interior points are discarded, so you can't dent/hollow a hull directly.
    # base = convex_hull("base", [
    #     (-0.5, -0.5, -0.55), (0.5, -0.5, -0.55),
    #     (0.5,  0.5, -0.55), (-0.5, 0.5, -0.55),    # wide bottom
    #     (-0.35, -0.35, -0.5), (0.35, -0.35, -0.5),
    #     (0.35,  0.35, -0.5), (-0.35, 0.35, -0.5)])  # narrower top -> a frustum
    # set_color(base, (0.2, 0.2, 0.22), roughness=0.6)

    # --- color the parts (render is RGB — match the object's colors) ---------
    set_color(body, (0.75, 0.15, 0.12), roughness=0.35)    # glossy red ceramic
    set_color(handle, (0.75, 0.15, 0.12), roughness=0.35)


# Placement — shared SCALE + REFERENCE_FRAME + JOINTS + per-frame FRAMES.
# What each field means and how it composes: conventions/POSE.md (SCALE / pose)
# and conventions/JOINTS.md (JOINTS / joint states). Below are just this mug's values.

SCALE = 1.0                 # a real mug is ~0.1 m; here 1 canonical unit reads fine

REFERENCE_FRAME = "frame0"  # the one frame in this single-image example

JOINTS = []                 # a rigid mug — nothing articulates

FRAMES = {
    # The mug stands upright at the origin, so +Z (canonical up) must map to -Y
    # (image up) to face camera 0: a +90° rotation about X does that, which is the
    # quaternion (cos45°, sin45°, 0, 0). tz = 2.2 puts it a couple units ahead.
    "frame0": {
        "pose": {"quaternion": (0.707107, 0.707107, 0.0, 0.0),
                 "translation": (0.0, 0.0, 2.2)},
    },
}
