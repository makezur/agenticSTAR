"""lib — shared leaves for the analysis tools (no tool logic, no cycles).

  * rasters   — image/mask loading + silhouette extraction + keep-region
  * raster    — exact per-pixel z-buffer rasterizer for projected triangle meshes
  * panels    — panel drawing: labels, scaling, hstack, heatmaps, timeline
  * depth_obs — observed-depth pointmap loaders + resampling
  * glb       — GLB loading into the canonical (+Z up) frame
  * pose_read — the one ingest seam for a pose.json document, and its gates
  * io        — JSON read/write, frame-stem + per-frame filename builders, and
                the argparse blocks shared across tools
"""
