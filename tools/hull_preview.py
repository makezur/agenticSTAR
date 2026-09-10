#!/usr/bin/env python3
"""hull_preview.py — ad-hoc convex-hull verifier for the convex_hull() helper.

Run in the 'artscript' micromamba env:
  micromamba run -n artscript python tools/hull_preview.py --points "0,0,0.5 0.4,0.4,-0.3 -0.4,0.4,-0.3 0,-0.5,-0.3"
  micromamba run -n artscript python tools/hull_preview.py --points-file pts.json
  micromamba run -n artscript python tools/hull_preview.py --glb runs/foo/mesh/object.glb
  micromamba run -n artscript python tools/hull_preview.py --points "..." --html hull.html

Modes (pick ONE input source; add --html to any of them for an interactive view):

  POINTS mode (--points / --points-file): the authoritative visual check. Writes a
  throwaway RUN_DIR/scene.py that builds ONE part with the REAL convex_hull() helper
  (imported `from shapes import convex_hull` — the same shared helper the agent uses,
  so it can never drift from it), then drives the existing Blender harness to render a
  turntable and export a GLB, then runs check_watertight.py on that GLB. You read the
  printed turntable PNGs (one per orbit direction, named with the azimuth/elevation
  they were shot from) to confirm the shape, and the watertight verdict to confirm
  it's a closed solid. This exercises the EXACT production path, not a re-derivation.

  GLB mode (--glb): a numeric/text inspector for an already-exported GLB. Per part it
  reports watertightness + face/vert counts + a convexity flag (part volume vs. its own
  convex-hull volume). The harness renders scene.py, not arbitrary GLBs, so this mode
  does NOT render a Blender turntable — it's a text report (+ optional --html).

  --html PATH: write a SELF-CONTAINED interactive HTML you open in a browser and
  drag-to-rotate (three.js inlined by trimesh.viewer.scene_to_html — zero external
  loads, no server). For points input it shows the trimesh-computed hull (blue,
  semi-transparent) with the input points overlaid as red dots, so you can eyeball
  from ANY angle which points ended up on the hull vs. discarded inside it. For --glb
  it shows the exported parts. This is a quick shape sanity-check; the watertight
  GATE still comes from the Blender path in points mode.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RENDER_SH = os.path.join(REPO, "harness", "render.sh")
HARNESS = os.path.join(REPO, "harness")
# analysis/ is a package: run check_watertight as a module from harness/.
CHECK_WT = "analysis.check_watertight"

# harness/ on the path for the shared turntable filename grammar (this script runs
# from the repo root, not from harness/).
sys.path.insert(0, HARNESS)
from core import turntable_views  # noqa: E402


# --------------------------------------------------------------------------- #
# point parsing
# --------------------------------------------------------------------------- #
def parse_points(points_str, points_file):
    """Return a list of (x, y, z) floats from --points or --points-file.

    --points: whitespace-separated triples, each "x,y,z" (e.g. "0,0,1 1,0,0 ...").
    --points-file: JSON — either a bare list of [x,y,z] triples, or {"points": [...]}.
    """
    if points_file:
        with open(points_file) as f:
            data = json.load(f)
        pts = data["points"] if isinstance(data, dict) else data
        return [(float(a), float(b), float(c)) for a, b, c in pts]

    pts = []
    for tok in points_str.split():
        parts = tok.split(",")
        if len(parts) != 3:
            raise SystemExit(f"--points: '{tok}' is not an x,y,z triple")
        pts.append(tuple(float(v) for v in parts))
    return pts


# The throwaway scene imports the REAL convex_hull() from the shared `shapes`
# package (the same helper the agent uses), so this preview can never drift from
# it — harness/ is on sys.path when render.sh execs the scene inside Blender.
SCENE_TEMPLATE = '''\
"""hull_preview throwaway scene — one convex_hull part, upright, single frame."""

import bpy

from shapes import convex_hull


def build():
    hull = convex_hull("hull", {points!r})
    mat = bpy.data.materials.new("hull_mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.35, 0.55, 0.85, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
    hull.data.materials.append(mat)


SCALE = 1.0
REFERENCE_FRAME = "frame0"
JOINTS = []
FRAMES = {{
    "frame0": {{
        "pose": {{"rotation_euler": (1.5708, 0.0, 0.0),
                 "translation": (0.0, 0.0, 2.5)}},
    }},
}}
'''


def points_mode(points, run_dir, keep):
    scene_path = os.path.join(run_dir, "scene.py")
    views_dir = os.path.join(run_dir, "views")
    glb_path = os.path.join(run_dir, "mesh", "object.glb")
    wt_path = os.path.join(run_dir, "mesh", "watertight_report.json")
    os.makedirs(os.path.join(run_dir, "mesh"), exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)

    scene_src = SCENE_TEMPLATE.format(points=points)
    with open(scene_path, "w") as f:
        f.write(scene_src)
    print(f"[hull_preview] wrote throwaway scene: {scene_path}")
    print(f"[hull_preview] {len(points)} points -> convex_hull('hull', ...)")

    # 1) render turntable + export the GLB via the real Blender harness.
    render = subprocess.run(
        ["bash", RENDER_SH, scene_path, views_dir,
         "--views", "turntable", "--export", glb_path],
        cwd=REPO, capture_output=True, text=True)
    sys.stdout.write(render.stdout)
    sys.stderr.write(render.stderr)
    if render.returncode != 0:
        raise SystemExit(
            "[hull_preview] render.sh FAILED (a degenerate/coplanar point set makes "
            "convex_hull() raise — see the ValueError above).")

    # find the pass dir the wrapper announced, then LIST the turntable PNGs it
    # actually wrote. Globbed rather than reconstructed: the view set is a flag
    # (--turntable-views, and the names carry az/el).
    m = re.search(r"PASS (\S+) OUTPUT DIR: (\S+)", render.stdout)
    pass_dir = m.group(2) if m else views_dir
    tt = sorted(glob.glob(os.path.join(pass_dir, turntable_views.IMAGE_GLOB)))

    # 2) watertight gate on the exported GLB.
    print("\n[hull_preview] --- watertight check ---")
    # run as a module from harness/ (analysis is a package); absolutize the GLB /
    # report paths so the cwd change can't misresolve a relative run_dir.
    wt = subprocess.run(
        ["python", "-m", CHECK_WT, os.path.abspath(glb_path),
         "--out", os.path.abspath(wt_path)],
        cwd=HARNESS, capture_output=True, text=True)
    sys.stdout.write(wt.stdout)
    sys.stderr.write(wt.stderr)

    print("\n[hull_preview] ============================================")
    print(f"[hull_preview] exported GLB : {glb_path}")
    print(f"[hull_preview] watertight   : PASS={wt.returncode == 0} ({wt_path})")
    print(f"[hull_preview] OPEN THESE to eyeball the shape from {len(tt)} angles:")
    for p in tt:
        print(f"[hull_preview]   {p}")
    if not tt:
        print(f"[hull_preview]   (none found in {pass_dir} — the render reported "
              "success but wrote no turntable images)")
    if not keep:
        print("[hull_preview] (throwaway scene kept for inspection; rm -rf "
              f"{run_dir} to clean)")
    return 0 if wt.returncode == 0 else 1


# --------------------------------------------------------------------------- #
# GLB inspector (text report + convexity flag via manifold3d, scipy-free)
# --------------------------------------------------------------------------- #
def glb_mode(glb_path):
    import trimesh
    import manifold3d

    scene = trimesh.load(glb_path, process=False)
    if isinstance(scene, trimesh.Scene):
        parts = {n: g for n, g in scene.geometry.items()
                 if isinstance(g, trimesh.Trimesh)}
    elif isinstance(scene, trimesh.Trimesh):
        parts = {"mesh": scene}
    else:
        raise SystemExit(f"[hull_preview] could not read a mesh from {glb_path}")

    print(f"[hull_preview] GLB inspector: {glb_path}  ({len(parts)} part(s))")
    all_wt = bool(parts)
    for name, geom in parts.items():
        m = geom.copy()
        try:
            m.merge_vertices(merge_tex=True, merge_norm=True)  # undo glTF seams
        except TypeError:
            m.merge_vertices()
        wt = bool(m.is_watertight)
        all_wt = all_wt and wt

        # convexity: compare the part volume to its own convex hull's volume
        # (manifold3d hull; ratio ~1 = convex, < 1 = has a concavity/hole).
        convex_note = ""
        if wt:
            try:
                hull = manifold3d.Manifold.hull_points(np.asarray(m.vertices, float))
                hv = float(hull.volume())
                pv = float(m.volume)
                ratio = pv / hv if hv > 1e-12 else 0.0
                verdict = "convex" if ratio > 0.98 else f"CONCAVE (fills {ratio:.0%} of hull)"
                convex_note = f"  vol={pv:.4g} hull_vol={hv:.4g} -> {verdict}"
            except Exception as e:  # noqa: BLE001 — diagnostic only
                convex_note = f"  (convexity check skipped: {e})"

        print(f"[hull_preview]   {name}: watertight={wt} "
              f"faces={len(m.faces)} verts={len(m.vertices)} "
              f"euler={int(m.euler_number)}{convex_note}")

    print(f"[hull_preview] all parts watertight: {all_wt}")
    return 0 if all_wt else 1


# --------------------------------------------------------------------------- #
# interactive self-contained HTML (drag to rotate) — trimesh scene_to_html
# --------------------------------------------------------------------------- #
def _point_markers(points, colors=(235, 60, 60, 255)):
    """Small cube markers at each point (a trimesh PointCloud pulls in scipy and
    renders as hard-to-see specks; cubes read clearly and are robust)."""
    import trimesh

    pts = np.asarray(points, float)
    span = float(np.ptp(pts, axis=0).max()) if len(pts) else 1.0
    d = max(span * 0.03, 1e-3)                    # marker size ~3% of extent
    markers = {}
    for i, p in enumerate(pts):
        b = trimesh.creation.box(extents=(d, d, d))
        b.apply_translation(p)
        b.visual.face_colors = colors
        markers[f"pt_{i}"] = b
    return markers


def write_html(out_path, points=None, glb_path=None):
    """Write a self-contained interactive HTML (three.js inlined; no server/CDN).

    points -> trimesh-computed hull (blue, translucent) + input points (red cubes).
    glb_path -> the exported parts loaded straight from the GLB.
    """
    import trimesh
    from trimesh.viewer import scene_to_html

    geoms = {}
    if glb_path:
        loaded = trimesh.load(glb_path, process=False)
        scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    else:
        pts = np.asarray(points, float)
        hull = trimesh.Trimesh(vertices=pts).convex_hull   # scipy Qhull
        hull.visual.face_colors = [90, 140, 215, 170]      # translucent blue
        geoms["hull"] = hull
        geoms.update(_point_markers(pts))                  # red input points
        scene = trimesh.Scene(geoms)

    html = scene_to_html(scene)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[hull_preview] wrote interactive HTML: {out_path}")
    print(f"[hull_preview]   open in a browser (drag to rotate). "
          f"self-contained, {len(html) // 1024} KB, no server needed.")
    if points is not None:
        on_hull = len(trimesh.Trimesh(vertices=np.asarray(points, float))
                      .convex_hull.vertices)
        print(f"[hull_preview]   {on_hull}/{len(points)} input points landed on the "
              f"hull (the rest are interior and discarded).")


def main():
    ap = argparse.ArgumentParser(
        description="Verify the convex_hull() helper: hull a point set via the real "
                    "Blender harness (turntable + watertight gate), or inspect a GLB.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--points", help='whitespace-separated x,y,z triples, e.g. '
                                     '"0,0,0.5 0.4,0.4,-0.3 -0.4,0.4,-0.3 0,-0.5,-0.3"')
    g.add_argument("--points-file", help="JSON: [[x,y,z],...] or {\"points\":[...]}")
    g.add_argument("--glb", help="inspect an already-exported GLB (text report only)")
    ap.add_argument("--html", metavar="PATH",
                    help="also write a self-contained interactive HTML (drag to "
                         "rotate; open in a browser, no server needed)")
    ap.add_argument("--run-dir", default=os.path.join(REPO, "runs", "hull_preview"),
                    help="throwaway run dir for points mode (default runs/hull_preview)")
    ap.add_argument("--keep", action="store_true",
                    help="(points mode) keep the run dir quietly")
    args = ap.parse_args()

    if args.glb:
        if args.html:
            write_html(args.html, glb_path=args.glb)
        sys.exit(glb_mode(args.glb))

    points = parse_points(args.points, args.points_file)
    if len(points) < 4:
        raise SystemExit("[hull_preview] need >= 4 points for a closed solid "
                         f"(got {len(points)}).")
    if args.html:
        write_html(args.html, points=points)
    sys.exit(points_mode(points, args.run_dir, args.keep))


if __name__ == "__main__":
    main()
