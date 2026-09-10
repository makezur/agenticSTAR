"""
check_watertight.py — watertight / manifold report for the exported GLB.

Runs in the 'artscript' micromamba env, from harness/:
  micromamba run -n artscript python -m analysis.check_watertight \
      runs/mug/mesh/object.glb --out runs/mug/mesh/watertight_report.json

Reports per-part and combined geometry health. The concrete gate toward the
project goal of "watertight mesh parts". Exit code 0 if every part is
watertight, else 1 (so it can be used in a shell gate).
"""

import argparse
import json
import sys

import trimesh

from analysis.lib import glb as glb_lib
from analysis.lib.io import write_json


def clean(mesh):
    """Merge vertices split by the GLB exporter along normal/UV seams.

    glTF/GLB duplicates vertices wherever normals or UVs differ, which makes a
    genuinely closed surface look open to trimesh's default check. Merging
    across normals/UVs restores true topology before we judge watertightness.
    (The shared loader already merges; kept for callers handing raw meshes.)
    """
    m = mesh.copy()
    glb_lib.merge_seam_vertices(m)
    return m


def geom_report(name, mesh):
    mesh = clean(mesh)
    return {
        "name": name,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "bounds": mesh.bounds.tolist() if mesh.bounds is not None else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("glb", help="path to the exported .glb")
    p.add_argument("--out", default="", help="write JSON report to this path")
    args = p.parse_args()

    # The shared canonical-frame ingest (analysis.lib.glb): parts keyed by
    # SCENE-GRAPH NODE — the scene.py part names JOINTS and the
    # self-intersection report speak — with node transforms + the glTF Y-up
    # rotation baked in, so `bounds` reads in canonical axes like every other
    # report. Watertightness itself is rotation-invariant; the frame only
    # matters for the bounds.
    parts = glb_lib.load_parts(args.glb)
    combined = trimesh.util.concatenate(list(parts.values())) if parts else None

    report = {
        "glb": args.glb,
        "num_parts": len(parts),
        "parts": [geom_report(n, m) for n, m in parts.items()],
    }
    if combined is not None:
        report["combined"] = geom_report("__combined__", combined)

    all_watertight = bool(parts) and all(p["is_watertight"] for p in report["parts"])
    report["all_parts_watertight"] = all_watertight
    report["pass"] = all_watertight

    text = json.dumps(report, indent=2)
    if args.out:
        write_json(args.out, report)
        print(f"[check_watertight] wrote {args.out}")
    print(text)
    print(f"[check_watertight] PASS={all_watertight}  parts={len(parts)}")
    sys.exit(0 if all_watertight else 1)


if __name__ == "__main__":
    main()
