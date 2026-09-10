# `match` view — the core comparison render (per frame)

**Answers:** "does my build + per-frame pose + joint state overlay each photo?"

For each requested frame, renders the object — posed by that frame's
`FRAMES[...]["pose"]` (shared `SCALE`) with its joint states applied — from the
**fixed camera 0** (identity extrinsic) using that frame's intrinsics (the tracking `K[k]`,
or an iPhone-13 placeholder without `--tracking`), to `match_<frame>.png`. You change
the match by editing that frame's pose / joint state (or the shape in `build()`) —
never by moving the camera. Every frame renders from camera 0; the harness moves
the OBJECT into each frame's camera (the known relative camera transform).

## Run
```bash
harness/render.sh RUN_DIR/scene.py RUN_DIR/views \
    --views match,turntable --tracking CAPTURE/tracking --frames 000000.jpg,000080.jpg \
    --ref-frame 000000.jpg --match-res <ref image> --samples 64 \
    --export RUN_DIR/mesh/object.glb
```
- **Always render at the source resolution:** pass `--match-res <ref image>` so
  `match_<frame>.png` overlays the photo and mask 1:1.
- **Engine:** defaults to EEVEE (fast) for iterating. Add `--engine CYCLES`
  `--samples 64` for the final color judgment and the exported GLB.
- **Frames / cameras:** `--tracking DIR --frames a,b --ref-frame a` supplies the known
  per-frame cameras. Without `--tracking`, a single frame is rendered with the
  iPhone-13 placeholder intrinsics.
- `--debug-project x,y,z` prints where a *posed* world point lands in pixels for
  the reference frame (sanity-check pose/intrinsics).

## Outputs (in the pass dir `render.sh` prints)
Each `render.sh` call writes into a fresh `RUN_DIR/views/<NNNN>/` pass dir and
PRINTS that dir; read this run's outputs from the printed dir. A view not rendered
this pass is absent from it (no stale leftovers).
- `match_<frame>.png` — one render per frame; compare each to its source. A
  correct build + per-frame pose + joint state overlays each photo 1:1.

`pose.json` (shared `scale`/`reference_frame`/`parts`/`joint_defs` + a `frames`
map of per-frame `object_pose` + `joints` + `view_index`/`camera_c2w`) is written
next to the GLB when `--export` is given, else into the pass dir
(`PASS_DIR/`, where PASS_DIR = the `views/<NNNN>/` dir `render.sh` printed). It is
the downstream (URDF) handoff.
