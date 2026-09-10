"""glb.py — the ONE ingest of an exported object.glb into CANONICAL-frame parts.

THE FRAME SEAM THIS MODULE OWNS. glTF mandates +Y up (spec §3.4: "glTF defines
+Y as up"), so Blender's exporter (rig/export.py, default export_yup=True)
bakes a -90 deg X rotation into every node: p_gltf = (x, z, -y)_canonical. The
harness's canonical frame is Blender's +Z up (pose.json "canonical"), and
EVERYTHING in pose.json — base poses, joint origins/axes, canonical units — is
authored there. So any consumer that combines GLB geometry with pose.json
quantities must first bring the mesh back:

    p_canonical = GLTF_TO_CANONICAL @ p_gltf        i.e. (x, -z, y)

Python consumers get canonical meshes FROM THE LOADER and never see the glTF
frame at all. Do NOT "fix" this at the exporter instead: a Z-up
GLB silently violates the spec, renders tipped over in every stock viewer, and
is indistinguishable from the Y-up GLBs in every existing run (glTF has no
field to declare an up-axis).

Pure trimesh + numpy — importable in the artscript env and Blender's python.
"""

import numpy as np
import trimesh


# glTF (+Y up) -> canonical (+Z up): (x, y, z)_gltf -> (x, -z, y). The exact
# inverse of the rotation Blender's export_yup bakes in.
GLTF_TO_CANONICAL = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

# canonical -> glTF, for WRITERS that must produce a spec-correct Y-up GLB
# outside Blender (the test fixtures use it so they match production files).
CANONICAL_TO_GLTF = GLTF_TO_CANONICAL.T.copy()   # pure rotation: inv == T


def merge_seam_vertices(mesh):
    """Merge vertices the GLB exporter split along normal/UV seams, in place.

    glTF duplicates vertices wherever shading differs, which makes a genuinely
    closed surface look open to trimesh's watertight check — merging across
    normals/UVs restores true topology before any judgement on it."""
    try:
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
    except TypeError:  # older trimesh without those kwargs
        mesh.merge_vertices()


def load_parts(glb_path):
    """{part_name: CANONICAL-frame trimesh} from an exported GLB.

    Names come from the SCENE-GRAPH NODES, not the geometry names: the exporter
    names geometries after Blender datablocks (`Cylinder.004`) while the nodes
    carry the scene.py part names the JOINTS refer to. Each node's own
    transform AND the exporter's Y-up rotation are baked in, so the returned
    meshes are in the canonical frame directly — safe to combine with
    pose.json poses and joint origins/axes with no further conversion.

    A part is its ROOT-LEVEL node plus every geometry descendant, merged into
    ONE mesh. That closes the two ways a multi-material part loses its name:

      * glTF splits a mesh into one primitive per material, and trimesh
        explodes an unmerged multi-primitive node into synthetic hex-suffixed
        children (`base` -> `base_11ca75`, ...) whose names CHANGE on every
        load. merge_primitives=True keeps such a part one geometry under its
        own node. The merged mesh carries a trimesh MultiMaterial (per-face
        `visual.face_materials` survives; there is no single baseColorFactor).
      * A file may also carry real geometry children under a named parent
        node; those are concatenated (ordered by their load-stable GEOMETRY
        name) into the parent's part.

    Vertices are merged across normal/UV seams (merge_seam_vertices) so a
    closed surface reads watertight.
    """
    scene = trimesh.load(glb_path, process=False, merge_primitives=True)
    if isinstance(scene, trimesh.Trimesh):
        mesh = scene.copy()
        merge_seam_vertices(mesh)
        mesh.apply_transform(GLTF_TO_CANONICAL)
        return {"mesh": mesh}

    graph = scene.graph
    geometry_nodes = set(graph.nodes_geometry)
    children = {}
    for parent, child, _attr in graph.to_edgelist():
        children.setdefault(parent, []).append(child)

    def sort_key(node):
        # geometry names are written by the exporter and load-stable; node
        # names of exploded children are regenerated per load.
        return (graph[node][1] if node in geometry_nodes else "", node)

    def descend(node, seen):
        # `seen` guards a malformed graph (a self/cyclic edge would recurse
        # forever) — a well-formed glTF node hierarchy is a tree.
        if node in seen:
            return []
        seen.add(node)
        found = [node] if node in geometry_nodes else []
        for child in sorted(children.get(node, ()), key=sort_key):
            found.extend(descend(child, seen))
        return found

    parts = {}
    seen = {graph.base_frame}
    for top in children.get(graph.base_frame, ()):
        pieces = []
        for node in descend(top, seen):
            T, geom_name = graph[node]
            geom = scene.geometry.get(geom_name)
            if not isinstance(geom, trimesh.Trimesh):
                continue
            mesh = geom.copy()
            mesh.apply_transform(GLTF_TO_CANONICAL @ np.asarray(T, dtype=float))
            pieces.append(mesh)
        if not pieces:
            continue
        mesh = pieces[0] if len(pieces) == 1 else trimesh.util.concatenate(pieces)
        merge_seam_vertices(mesh)
        parts[top] = mesh
    return parts
