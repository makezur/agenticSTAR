# `ids` view — "which primitive is which"

**Answers:** "which part owns this region of the image?"

An ID map from the match camera: every part in your scene is flat-colored with a
distinct unlit color, so you can see which primitive produced which pixels and
know exactly which one to adjust in `scene.py`. It covers **every** part, however
you named it (handy when you spawn many hard-to-name parts just for fidelity).

## Run
```bash
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views match,ids \
    --match-res IMAGE
```

## Outputs (in the pass dir `render.sh` prints)
Each `render.sh` call writes into a fresh `RUN_DIR/views/<NNNN>/` pass dir and
PRINTS that dir; read this run's outputs from the printed dir. A view not rendered
this pass is absent from it (no stale leftovers).
- `ids.png` — the dense flat-color map (overlays `match.png` 1:1).
- `ids.txt` — human legend: `idx  name  RGB  pixels  status`.
- `ids.json` — the same legend as data, plus each part's `centroid` and `occluded`.

`Read` `ids.png` next to `ids.txt` to map a colored region → the part's
`scene.py` name. A part fully hidden in the match view is still listed
(`OCCLUDED`, zero pixels) — check the turntable or `visibility` views to see it.
The ID pass restores materials afterward, so it does **not** affect `match.png`,
the turntable, or the GLB.
