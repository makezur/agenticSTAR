"""export.py — optional voxel remesh + canonical GLB export.

The parts are restored to canonical (see scene.restore_world) before this runs,
so the exported mesh is the CANONICAL rest pose — the per-frame poses + joint
states live in pose.json only.
"""

import json
import os
import struct

import bpy


def maybe_remesh(remesh_arg):
    """If remesh 'voxel:S' with S>0, add+apply a voxel remesh to each part."""
    if not remesh_arg or ":" not in remesh_arg:
        return
    kind, val = remesh_arg.split(":", 1)
    try:
        size = float(val)
    except ValueError:
        return
    if kind != "voxel" or size <= 0:
        return
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        bpy.context.view_layer.objects.active = obj
        mod = obj.modifiers.new("rig_remesh", "REMESH")
        mod.mode = "VOXEL"
        mod.voxel_size = size
        bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"[render_wrapper] applied voxel remesh size={size}")


def glb_node_names(path):
    """Node names from a GLB's JSON chunk — stdlib only (runs in Blender's
    python, where trimesh may be absent)."""
    with open(path, "rb") as f:
        header = f.read(20)
        if header[:4] != b"glTF":
            raise ValueError(f"not a GLB: {path}")
        chunk_len = struct.unpack("<I", header[12:16])[0]
        tree = json.loads(f.read(chunk_len))
    return {n.get("name") for n in tree.get("nodes", []) if n.get("name")}


def check_exported_parts(path, part_names):
    """Fail the pass if a part shipped without its name.

    pose.json's `parts` (and every JOINTS `child`) refer to GLB nodes BY NAME;
    a part the exporter dropped or renamed poisons every downstream consumer
    (metrics, self-intersection, viewers), so assert it at write time, where
    the fix (the scene) is still in hand.
    """
    missing = sorted(set(part_names) - glb_node_names(path))
    if missing:
        raise RuntimeError(
            f"exported GLB {path} has no node for part(s): "
            f"{', '.join(missing)} — the exporter dropped or renamed them; "
            "pose.json names parts by GLB node, so downstream consumers "
            "would silently lose this geometry.")


def export_glb(path, remesh_arg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    maybe_remesh(remesh_arg)
    # Export only mesh parts (camera/lights excluded via use_selection).
    bpy.ops.object.select_all(action="DESELECT")
    part_names = []
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            obj.select_set(True)
            part_names.append(obj.name)
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", use_selection=True,
    )
    check_exported_parts(path, part_names)
    print(f"[render_wrapper] exported {path}")
