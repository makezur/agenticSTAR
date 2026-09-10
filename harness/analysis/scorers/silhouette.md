# `scorers/silhouette.py` — silhouette IoU + RGB residual (the numeric gate)

**Primary quantitative signal (when a MASK is provided): the rendered silhouette
must COINCIDE with the mask, 1:1.** The render is on a transparent film, so its
alpha channel is an exact silhouette. Because every frame renders from the fixed
camera 0 with the object posed at true scale (the frame's `pose` + shared
`SCALE`), the render overlays the photo directly — so the primary metric is the
**raw pixel IoU** (`iou_raw`), which counts placement, scale, AND proportions.

## Run (artscript env)
```bash
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.scorers.silhouette \
    --mask MASK --render PASS_DIR/match.png --source IMAGE \
    [--hand-mask HAND_MASK] --out RUN_DIR/mesh/metrics.json
```
(PASS_DIR = the `views/<NNNN>/` dir `render.sh` printed.)
Exit code is the gate: `0` iff `report[gate_field] >= --pass-iou` (default
`0.98`), where `gate_field` is `iou_visible` when a hand mask is given, else
`iou_raw`.

## Key fields
- **`iou_raw`** — raw 1:1 IoU over the **full image**. The primary metric and
  gate **when no hand mask is given**; still reported (as a diagnostic) when one
  is. Drive iteration to raise the gate metric toward 1.0 (target ≥ **0.98**).
- `aspect_ratio_source` vs `aspect_ratio_render` — proportion **hint**: width/height
  of each tight silhouette bbox. A mismatch is usually a shape/`build()` issue (a
  uniform `scale` doesn't change aspect). It is a hint, not a shape-only score.
- `gate_field` — names the metric the gate/exit-code uses (`iou_raw` or
  `iou_visible`), so you always know which number you're driving.

**IoU is the best, most precise signal in the LOCAL NEIGHBOURHOOD.** Once the config
is already close, nothing beats `iou_raw` for pinning the exact pose / scale / joint
value: it is exact about 2D silhouette overlap for this frame, sub-pixel and
deterministic. That local precision is why it's the gate and why [`sweep`](../../views/sweeps/sweep.md)
optimizes it to refine.

**But that precision is LOCAL and blind — it does not tell you shape vs. pose.** It is
a single silhouette number and is **depth-blind** (a nearer+smaller pose paints the
same outline), so a low `iou_raw` alone cannot say whether the cause is geometry or
placement, and it is meaningless when you're in the *wrong* neighbourhood (far off, or
the geometry itself is wrong). There is deliberately **no** bbox-normalized
"shape-only" IoU — a normalized number invited deciding shape from a metric instead of
from the signals that actually see it. Make the shape/pose/joint call from the
**turntable** (coherence, per state), direct source/render comparisons, the
**aspect-ratio hint** above, and the **per-frame
IoU spread** (a defect in *every* frame ⇒ shape; a low score in *one* frame ⇒ pose).
See AGENT_TASK.md "Shape vs. pose vs. articulation".

## Hand occlusion — `--hand-mask` (the object continues behind the hand)
When the photo has a **hand occluding the object**, pass a binary hand
(occluder) mask. The hand region becomes a **don't-care**: every metric is
scored only over the keep-region **K = ¬hand**, so the render is neither
penalized nor rewarded for what it does behind the hand (the object legitimately
continues there, and SAM labels those pixels as hand, not object).
- **`iou_visible`** — `|R∩M∩K| / |(R∪M)∩K|`. The **primary gate** whenever a
  hand mask is present (`gate_field="iou_visible"`). Always `≥ iou_raw`, and
  equals `iou_raw` when the hand mask is empty. Note it is *not* a free lunch:
  ignoring pixels where the render already **agrees** with the mask lowers it —
  you only gain back genuinely-occluded false positives, so a too-large hand
  mask hurts rather than helps.
- **`--hand-dilate`** (default **0**, the hand mask exactly) — grow the ignore-region by this
  fraction of the longer image side (elliptical kernel) before ignoring, to
  absorb the thin rim where the SAM hand boundary doesn't line up with the true
  occlusion edge. `0` = use the hand mask exactly. `hand_coverage` reports the
  final ignored fraction.
- the aspect-ratio hints and the RGB residual (`overlap_frac`, `color_score`) are
  also restricted to K, so every secondary signal stays fair under occlusion.
- **Degenerate:** if K leaves no scorable pixels (object fully under the hand, or
  an over-dilated / bad hand mask), `iou_visible` is `null`,
  `visible_region_empty: true`, and the run **fails** the gate (never a silent
  1.0 pass).

## Secondary signal — RGB residual (only with both MASK and `--source`)
Measured feature-aligned over the silhouette overlap, so a shape mismatch is not
mistaken for a color error:
- `color_score` — `1 − mean_abs_RGB_error/255`, 0..1 (higher = closer color);
- `rgb_mae` — mean per-channel error in 0..255 (lower is better);
- `overlap_frac` — how much of the object the color score covers (low = the
  silhouettes barely overlap; fix shape/placement first).

**`color_score` is SECONDARY — only trust it once `iou_raw` is high (≳0.85).**
While silhouettes barely overlap, few pixels correspond, so the color number is
unreliable. Fix silhouette/placement FIRST.
