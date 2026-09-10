# `depth` view — render a per-pixel depth map (geometric supervision)

**Answers:** "does my posed geometry sit at the right DEPTH?" — the question
silhouette IoU is blind to.

Renders the object, posed by the frame's `pose` (+ shared scale + joint states),
from the **fixed camera 0** (identity extrinsic + the resolved per-frame
intrinsics), but writes the camera-space **planar depth**
(Z) of every pixel to `depth.npy` instead of an RGBA image. That depth is what
[`../analysis/scorers/depth.py`](../analysis/scorers/depth.md) compares against the capture's observed
pointmap's depth channel, so a part floating toward the camera — which reads as a
perfect silhouette overlap — finally shows up as a depth error.

Blender's Z pass is **planar** camera-space depth (distance along the view axis,
matching the observed `local[...,2]`), not euclidean ray length. Background / uncovered
pixels are stored as `+inf`, so the analysis side selects the object with a plain
`np.isfinite` regardless of the camera clip planes.

## Run
```bash
harness/render.sh RUN_DIR/scene.py RUN_DIR/views \
    --views depth --match-res IMAGE \
    --intrinsics "fx,fy,cx,cy,W,H"        # the observed view's K, scaled to render res
```
- **Pair it with the observed view's camera.** Use `--intrinsics` (or
  `--camera-json`) with the K of the tracking view whose frame is your source image.
  Scale the tracking K (stored at its own grid `Wp×Hp`) to the render resolution:
  `fx*=W/Wp, cx*=W/Wp, fy*=H/Hp, cy*=H/Hp`. Default camera 0 = identity extrinsic,
  which is exactly the frame the per-view observed `local` pointmap lives in.
- **Output is `.npy`, not `.exr`:** the analysis env's OpenCV cannot decode
  OpenEXR, so the view reads Blender's Z pass back through numpy and saves a
  float32 array that `np.load` opens with no codec dependency.
- **Cheap and non-destructive:** run it alongside `match`/`turntable` in one
  launch (`--views match,turntable,depth`). It snapshots and restores the
  compositor, the view-layer Z-pass flag, and the camera/render resolution, so
  `match.png`, the turntable, `pose.json`, and the exported GLB are unaffected.
- Works on both EEVEE (default) and Cycles.
- `--debug-project 0,0,Z` prints a posed point's pixel + camera depth — the value
  at that pixel in `depth.npy` should equal that `Z` (verifies the planar-depth
  convention).

## Multi-frame
Every frame renders from the **same** fixed camera 0 (identity extrinsic). Instead
of moving the camera to a non-reference view, the harness **moves the object** into
that view (`M_k = T_{k<-ref} @ M_ref`, the relative camera transform
`inv(c2w[k]) @ c2w[ref]`, plus the frame's joint states) and swaps in the frame's
intrinsics `K[k]`. So the rendered depth is already in frame-k's camera frame and
scores RAW against the observed `local[k]` — no per-view pose baked into the camera. One
unit canonical mesh + shared scale + per-frame poses is the same "build canonical,
then pose" model the harness already uses.

## Outputs (in the pass dir `render.sh` prints)
Each `render.sh` call writes into a fresh `RUN_DIR/views/<NNNN>/` pass dir and
PRINTS that dir; read this run's outputs from the printed dir. A view not rendered
this pass is absent from it (no stale leftovers).
- `depth_<frame>.npy` — float32 `(H, W)` planar camera-space depth for each frame;
  `+inf` where there is no geometry. Score it with
  [`../analysis/scorers/depth.py`](../analysis/scorers/depth.md).
