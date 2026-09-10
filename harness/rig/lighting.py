"""lighting.py — neutral 3-light rig + fallback material for unmateraled parts."""

import bpy
from mathutils import Vector

from rig.camera import CAM0_UP, orbit_offset, aim_camera


def setup_lighting(center, radius):
    """Neutral, consistent 3-light setup + soft gray world so shape reads clearly."""
    world = bpy.data.worlds.new("rig_world")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        # Neutral gray ambient so colors/textures read true. The PNG background
        # itself is transparent (film_transparent), so this only lights the object.
        bg.inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
        bg.inputs[1].default_value = 0.6
    bpy.context.scene.world = world

    def add_sun(name, azimuth, elevation, energy):
        data = bpy.data.lights.new(name, "SUN")
        data.energy = energy
        obj = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(obj)
        # Light DIRECTION expressed in the camera-0 frame (-Y up), so the key
        # comes from above-front regardless of where the object is built. A sun
        # points along its local -Z; aim that down the incoming light ray.
        d = orbit_offset(1.0, azimuth, elevation)  # from target toward the sun
        aim_camera(obj, d, Vector((0.0, 0.0, 0.0)), up=CAM0_UP)
        return obj

    add_sun("key", azimuth=-35, elevation=55, energy=4.0)
    add_sun("fill", azimuth=160, elevation=25, energy=1.5)
    add_sun("rim", azimuth=60, elevation=40, energy=2.0)


def apply_default_material():
    """Fallback material ONLY for parts the agent left unmaterialed.

    The agent is expected to color/texture parts in scene.py (render is RGB), so
    any material it assigned is preserved untouched. Parts with no material get a
    light off-white so they still read cleanly rather than Blender's flat default.
    """
    mat = bpy.data.materials.get("rig_fallback")
    if mat is None:
        mat = bpy.data.materials.new("rig_fallback")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (0.82, 0.82, 0.85, 1.0)
            bsdf.inputs["Roughness"].default_value = 0.5
    for obj in bpy.data.objects:
        if obj.type == "MESH" and not obj.data.materials:
            obj.data.materials.append(mat)
