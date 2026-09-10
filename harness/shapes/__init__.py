"""shapes — the shared BUILD-TIME helpers every scene.py imports.

These are the boilerplate mesh/material helpers an agent-authored scene.py uses
inside build() to make canonical geometry. They are shared infrastructure, so
IMPORT them — do NOT redefine them in your scene.py:

    from shapes import box, convex_hull, set_color, apply_boolean

The API:
  box(name, center, size)          — axis-aligned cube of FULL extent `size`
                                      (sx, sy, sz), centered at `center`.
  convex_hull(name, points)        — a single CONVEX solid = the convex hull of a
                                      point set; inherently watertight. CONVEX ONLY
                                      (carve concavities afterward with apply_boolean).
  set_color(obj, rgb, ...)         — give `obj` a Principled BSDF material.
  apply_boolean(target, cutter,    — apply a boolean modifier (DIFFERENCE by
               operation=...)         default) and remove the cutter object.

Everything is authored in the CANONICAL frame: +Z up, object centered at the
origin, longest dimension ~= 1 unit. Objects created here are linked into the
active scene collection so build_from() (harness/rig/scene.py) collects them into
the 'parts' collection like any primitive.

This package sits on the Blender side (it imports bpy/bmesh), alongside rig/ and
views/. harness/ is on sys.path at exec time (see render_wrapper.py), so scenes
run via runpy.run_path can `from shapes import ...` directly.
"""

import bpy
import bmesh


def set_color(obj, rgb, roughness=0.5, metallic=0.0):
    """Give `obj` a Principled BSDF material of the given color.

    rgb: (r, g, b) in 0..1. Keep it simple; for real textures build an image
    texture node graph instead. The render is RGB, so color matters.
    """
    mat = bpy.data.materials.new(f"{obj.name}_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def apply_boolean(target, cutter, operation="DIFFERENCE"):
    """Apply a boolean modifier and remove the cutter object."""
    mod = target.modifiers.new(name="bool", type="BOOLEAN")
    mod.operation = operation
    mod.object = cutter
    mod.solver = "EXACT"
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def box(name, center, size):
    """Axis-aligned box: full extent `size` = (sx, sy, sz), centered at `center`.

    primitive_cube_add(size=1.0) ALREADY spans +/-0.5 (full extent 1.0), so scale
    by the FULL size you want — NOT size/2. (Blender's *default* cube spans +/-1,
    which is where the tempting size/2 comes from; with size=1.0 it is wrong and
    halves every box.) Halving boxes while part centers use full dimensions is the
    classic way parts detach from the body — see harness/shapes/README.md.
    """
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    o = bpy.context.active_object
    o.name = name
    o.scale = (size[0], size[1], size[2])   # full size — NOT size/2
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return o


def convex_hull(name, points, recalc=True):
    """Build a single CONVEX solid = the convex hull of `points`.

    points: iterable of (x, y, z) in the CANONICAL frame. Returns the new mesh
    object (already linked to the scene, so the harness collects it into the
    'parts' collection like any primitive — see build_from() in harness/rig/scene.py).

    Use this for convex shapes the cube/cylinder/sphere/cone/torus set can't
    express directly: wedges, prisms, tapered/faceted blocks, trapezoidal bodies.
    The hull of a point set is a closed convex polytope, so it is INHERENTLY
    watertight/manifold (it passes check_watertight.py on its own).

    CONVEX ONLY. A hull can never have a dent, a hole, a concave notch, or an
    internal cavity — it shrink-wraps the points, so any point you place INSIDE
    the hull is simply discarded. To get a concavity, build the convex blank with
    this helper and then CARVE it with a boolean DIFFERENCE cutter (see
    apply_boolean); for a ring, use the torus primitive. Do NOT try to fake
    concavity by adding interior points — they are dropped.

    REQUIRE >= 4 points that are NOT all coplanar and NOT all collinear. A
    flat/degenerate set yields an open zero-volume sheet that is NOT watertight
    and fails the watertight gate; this helper raises early if that happens.
    """
    bm = bmesh.new()
    for p in points:
        bm.verts.new((float(p[0]), float(p[1]), float(p[2])))
    bm.verts.ensure_lookup_table()

    res = bmesh.ops.convex_hull(bm, input=bm.verts, use_existing_faces=False)

    # Delete the verts NOT on the hull. NOTE: a vert can appear in BOTH
    # geom_interior and geom_unused, and bmesh.ops.delete rejects duplicates
    # ("found the same ... used multiple times") — dedup (order-preserving) first.
    # Deleting the leftover VERTS cascades to their edges/faces, leaving no loose
    # geometry.
    leftover, seen = [], set()
    for g in res["geom_interior"] + res["geom_unused"]:
        if g not in seen:
            seen.add(g)
            leftover.append(g)
    if leftover:
        bmesh.ops.delete(bm, geom=leftover, context="VERTS")

    bm.faces.ensure_lookup_table()
    n_faces = len(bm.faces)
    if n_faces < 4:
        bm.free()   # free BEFORE raising; the count is captured above (freed bm
                    # can't be read, or the message itself would ReferenceError).
        raise ValueError(
            f"convex_hull('{name}', ...): degenerate hull ({n_faces} faces). "
            f"Need >= 4 points that are NOT coplanar / collinear to make a "
            f"closed solid.")

    if recalc:                       # convex_hull already winds outward; safeguard
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()

    me = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)   # so build_from() collects it
    return obj
