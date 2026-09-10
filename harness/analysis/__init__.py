"""analysis — out-of-Blender scoring & comparison tools (artscript env).

Runs in the `artscript` micromamba env (numpy/opencv/trimesh), NOT inside
Blender. Layered as a downward DAG so the pieces stay coherent:

    viz  →  scorers  →  rollup / temporal  →  lib  →  core

  * `lib/`     — shared leaves with NO tool logic: pose.json ingest + its
                 state_json.md gates (`pose_read`), GLB ingest — the ONE
                 glTF-Y-up -> canonical-Z-up seam (`glb`), raster/mask I/O
                 (`rasters`), panel drawing (`panels`), observed-depth loaders
                 (`depth_obs`),
                 JSON + filename conventions + shared argparse blocks (`io`).
  * `scorers/` — per-frame scoring: `silhouette` (the IoU gate), `depth`
                 (rendered-vs-observed depth agreement), `critic` (VLM judge).
  * `viz/`     — visualization panels: `composite`, sheets, `sweep_sides`.
  * `rollup/`  — cross-frame roll-up: `aggregate`.
  * `temporal/` — motion through time on top of `pose_diff`: `report` (THE entry
                 point — one row per frame, all scalars, table + JSON),
                 `sequence` (the producer: per-step residuals + the per-frame
                 track), `derivatives` (velocity/acceleration and their radial
                 component), `seams` (which refiner posed which frame).
  * `pose_diff` — the core two-pose maths + the per-frame placement lift
                 (top level; `temporal/` builds on it).
  * `check_watertight` — standalone GLB gate (top level).
  * `measure_depth` — measure tentative per-frame object poses off the observed
                 pointmap (translation + camera-motion rotation; top level).

Tools are invoked as modules from `harness/`:
    micromamba run -n artscript python -m analysis.viz.composite ...
"""
