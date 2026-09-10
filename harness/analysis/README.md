# `analysis/` — out-of-Blender scoring & comparison tools

These run in the **`artscript` micromamba env** (numpy/opencv/trimesh), NOT
inside Blender. `analysis/` is a Python **package**, layered as a downward DAG so
each piece has one job:

```
viz  →  scorers  →  lib  →  core
```

Invoke a tool as a module, from `harness/` (or from the repo root with
`PYTHONPATH=harness`):

```
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.<pkg>.<tool> ...
```

Each tool has a `.md` next to its `.py` — that doc is the single source of truth.

## `lib/` — shared leaves (no tool logic)
| module | role |
|---|---|
| `lib/rasters.py` | image/mask loading, silhouette extraction, keep-region (`load_bgr`, `load_mask`, `render_silhouette`, `build_keep`, `load_rgba`) |
| `lib/panels.py` | panel drawing + the silhouette-overlap COLOURS (`OVL_*`, `OVERLAP_LEGEND`, `overlay_panel`, `overlap_legend`) — one definition shared by every tool that paints or captions that panel; plus `Panel`/`label_row`/`label` (header bar **stacked above** the image, at one height per ROW — `BAR_HEIGHT`, or `CAPTIONED_BAR_HEIGHT` when some column's caption cannot ride on its title line (`sub_fits_title_line`) — so nothing is covered and panels of equal height line up row-for-row), `solid`, `scale_to_height`, `hstack_panels`, `vstack_rows`, `turbo_heatmap`, `signed_diverging_heat`, `timeline_save` |
| `lib/depth_obs.py` | OBSERVED-depth pointmap loaders + resampling (`resolve_view`, `load_view`, `load_c2w`, `relative_pose`, `resample_to`). Named for the role, not the backend — `pi3x` survives only as a `depth_config.json` `backend` value. Our OWN render's depth is `lib/rasters.load_render_depth` |
| `lib/io.py` | JSON read/write + the per-frame filename contract (`read_json`, `write_json`, `frame_stem`, `metrics_json`/`depth_json`) |
| `lib/pose_read.py` | the one **ingest** seam for a pose.json document (`load_pose_json`, `frame_entry`, `frame_names`), and therefore the one place [state_json.md](../../conventions/state_json.md) is enforced on read: §1 quaternion+translation present, §2 `joints` a sibling and never nested in the pose, §8 a missing rotation RAISES rather than defaulting to identity (identity is a plausible-looking pose, so the residual would come out large and legible instead of failing). Derived caches (`matrix4x4`/`rotation_euler`) warn — a stale writer, not a bad read. Re-exported by `pose_diff`, so every reader gets the gates |

## `scorers/` — per-frame scoring
| tool | role | doc |
|---|---|---|
| `scorers/silhouette.py` | per-frame silhouette IoU (`iou_raw` primary gate) + aspect-ratio hint + RGB residual — `viz.composite --metrics-out` runs it for you, so you rarely call it directly. Writes `metrics_<stem>.json` | [silhouette.md](scorers/silhouette.md) |
| `scorers/depth.py` | per-frame depth/pointmap agreement vs an observed view (`--view N`; the depth-blindness cure for IoU — score it every iteration when Pi3X is present). Writes `depth_<stem>.json` | [depth.md](scorers/depth.md) |

There is deliberately no temporal *scorer*. Judging whether a step's motion is
real is not a measurement — it is a look at two frames and their renders, and its
output is a written verdict with an owner. That lives with the process, in
[multiagent/calls.py](../multiagent/calls.py) (the vocabulary) and
`multiagent.windows selfcheck` / `adjudicate` (who calls what). `temporal/` here
measures and reports; it never decides.

## `viz/` — visualization panels
| tool | role | doc |
|---|---|---|
| `viz/rows.py` | the **one** comparison-row builder + its column registry (`source`, `render`, `overlap`, `masked_source`, `depth`, `depth_residual`; `match` and `silhouette` are accepted aliases for `render`/`overlap`). `composite`, `sweep_sides` and `candidate_sheet` all build their rows here, so a column tuned once is tuned everywhere, and the default column ORDER is decided once (`DEFAULT_COLUMNS`); add a column by adding one registry entry | module docstring |
| `viz/composite.py` | the `[source \| render \| silhouette]` visual judge panel (per frame) — the only viz that also SCORES (metrics JSON, depth, timeline) | [composite.md](viz/composite.md) |
| `viz/frame_sheet.py` | paginated, temporally ordered initial source-frame sheets with readable IDs, automatic page splitting, subset/range/glob selection, and an auditable manifest. The grid flags are a page-footprint BUDGET, not a grid to fill: a short selection (a five-frame window sheet) is repacked into the same footprint with a rechosen grid and bigger tiles instead of black filler cells, and pages are balanced so the last one is not the only sparse one — clamped to native resolution, never below `--tile-height`, `--no-pack` for the literal grid | [frame_sheet.md](viz/frame_sheet.md) |
| `viz/pose_pair_sheet.py` | standalone adjacent-frame pose diagnostic: content-addressed software renders from the committed canonical GLB + pose JSON, opaque source overlays with a depth-aware object-base XYZ triad, and one two-frame sheet per temporal pair. Managed-run output belongs to the active iteration; rendering, caching, provenance, and pair bookkeeping are internal | [pose_pair_sheet.md](viz/pose_pair_sheet.md) |
| `viz/depth_units.py` | OUR rendered depth in **OBJECT UNITS** (depth / the predicted scale `s`) — GT-free, so it is the depth visual for monocular runs (`depth_config.json` backend `none`). Provides the standard row's `DEPTH` panel (NEAR/FAR key from the same turbo LUT as the pixels) and the **depth sheet**: every frame's heatmap on ONE shared NEAR/FAR scale, paged by `frame_sheet`'s grid, where poses breathing in depth read as tiles pulsing blue↔red across time | [depth_units.md](viz/depth_units.md) |
| `viz/gap_sheet.py` | the capture's own DENSE frames inside one temporal step (`--step A:B` between two of the run's keyframes) — the plausibility escalation for a residual that stays ambiguous from its two endpoints. Source only (the in-between frames have no poses, so no renders exist); default stride fits one page with both endpoints kept and the stride printed, `--every` overrides; one sheet dir per step under `RUN_DIR/gap_sheets/`, drawn entirely by `frame_sheet` | [gap_sheet.md](viz/gap_sheet.md) |
| `viz/turntable_sheet.py` | **one page per articulation state** from a render pass's turntable orbit views — opened by the state's SOURCE photo as a reference tile (identity anchor, not a comparison column), tiles labelled with the direction each was shot from, grouped FROM ABOVE / LEVEL / FROM BELOW, with the state's source frame in the header and an auditable per-tile manifest. What you read for the 3D coherence gate instead of N loose PNGs; the shape pass builds it every pass | [turntable_sheet.md](viz/turntable_sheet.md) |
| `viz/candidate_sheet.py` | paginated sweep/apply candidate pages; defaults to `[SOURCE \| RENDER \| OVERLAP]` (`masked_source` is opt-in), with green/yellow/magenta raw-IoU overlap and an auditable candidate-ID manifest | [candidate_sheet.md](viz/candidate_sheet.md) |
| `viz/seam_sheet.py` | every window **seam** on one sheet, each as a visual PAIR band: the two frames' sources above their committed renders (`--pass-dir`), so a basin flip is visible. Not `frame_sheet`: a seam is TWO frames compared against each other, where the comparison is the whole content, so the pair gets a bright rim and a wide gutter and context is opt-in. Draws only — it attaches no verdict; `multiagent.windows adjudicate` prints the invocation for the seams you owe a call on | [seam_sheet.md](viz/seam_sheet.md) |
| `viz/sweep_sides.py` | one-candidate-per-strip side-by-sides for a sweep/apply order's rendered candidates (`sweep_best.png` + `--sweep-dump-topk` tops) — the silhouette-trap check against the source, no re-render. `pool/panels.py` builds these for every pool order with the sheet's shared row (`SOURCE \| RENDER \| OVERLAP`), so the silhouette is on the strip too; the CLI defaults to its legacy picture columns and stays usable post-factum on any past sweep dir | module docstring |

## `rollup/` — cross-frame
| tool | role | doc |
|---|---|---|
| `rollup/aggregate.py` | roll up per-frame `metrics_*`/`depth_*` into one report + the cross-frame moved-flag checks | [aggregate.md](rollup/aggregate.md) |

## `temporal/` — motion through time

**No thresholds in this package, and no verdicts.** Every tool here reports
magnitudes — often in an attention ORDER (biggest first), never as a filtered
list of problems. Nothing being small means anything was cleared, and nothing
being large means anything is wrong: only the pictures can tell a real half-turn
from a spurious flip. [maths.md](temporal/maths.md) is the reading guide for what
each number is. Judging a step is a separate act with a named owner and a written
verdict — see [multiagent/calls.py](../multiagent/calls.py) and
`multiagent.windows selfcheck` / `adjudicate`.

| tool | role | doc |
|---|---|---|
| `temporal/report.py` | **THE entry point.** One row per frame, every value a single scalar: velocity + acceleration of the base pose (rotation, translation, radial) and of each joint. Prints an aligned table (~4x cheaper than JSON, and the only form you can scan down a column) and writes the same numbers as JSON via `--out`. Marks window **seams** given a `--run-dir`, and warns loudly on non-consecutive frames | [report.md](temporal/report.md) |
| `temporal/sequence.py` | per-step residuals along the frame timeline (`diff_sequence`; each consecutive pair diffed via `pose_diff`, path-length summary) **plus** the per-frame placement track and its `derivatives` block. Every frame must carry a camera pose or none may — partial coverage raises, since the two frames of reference are not comparable. Run it on the **current** `mesh/pose.json` before a windows round, or under `report`, which is the form a reader wants | [maths.md](temporal/maths.md) |
| `temporal/seams.py` | which refiner posed which frame, and where two windows meet (`crossing_steps`, `step_key`, `step_lines`, `rotation_order`). Pure **ownership** bookkeeping — consults no magnitude, because a seam matters for *who* posed it, not how big it is |
| `temporal/derivatives.py` | the derivative layer over the per-frame track: velocity, acceleration (rotation's as the increments' **ratio**, not a difference), and each one's signed **radial** component about the camera→object direction (+ = moving away). Read radial beside the row's own speed — a radial ≈0 alone cannot tell "moved across the view" from "did not move"; the across-view part is `sqrt(speed² − radial²)`, so it is not a separate field. Separate from the per-step residual because a derivative of a *magnitude* is not a derivative of the motion | [maths.md](temporal/maths.md) |

## top level
| tool | role | doc |
|---|---|---|
| `check_watertight.py` | per-part watertight/manifold gate on the exported GLB | [check_watertight.md](check_watertight.md) |
| `self_intersection.py` | per-frame part-pair **overlap volumes** (exact manifold3d booleans in the canonical frame — base-pose invariant, so only distinct joint configurations are computed). One row per frame (the temporal report's shape; `--pairs` for per-pair spans) with worst `frac_smaller`/`size` (cube-root overlap volume); onsets name the step that drove parts together (read beside the temporal report). Same-rigid-group overlaps are constant **design contacts**, stated once. Non-watertight parts excluded and NAMED. A **warning, not a gate** — `!!` is a display mark, nothing decides on it. Run by every `shape_pass.sh` pass | [self_intersection.md](self_intersection.md) |
| `measure_depth.py` | measure tentative per-frame object poses off the observed pointmap — a per-frame pose *prior* (translation measured per frame + rotation from the camera motion), plus a fused shared `SCALE`/proportions. Multi-frame `--run-dir`/`--frames` emits a paste-ready `pose_snippet` (pose-only fragments to paste INTO each `FRAMES` entry, so they can't clobber `moved`/`joints`); never decides `moved` (the agent's visual read does; aggregate cross-checks). Not a gate — no pass/fail | [measure_depth.md](measure_depth.md) |
| `pose_diff.py` | the core two-pose maths + the per-frame placement **lift** every temporal number is built from (`lift_placement`: `W = R_cam R_obj` and `offset_canon = R_cam t/s`). Compares committed poses of one articulated object — rotation as the geodesic angle of the body-frame increment `W_i^T W_j` in degrees, translation as the difference of the offsets (in object units and in scene units) + each joint's signed change with its declared `limit` beside it (`joint_deltas` is the one home for that math; `joint_states` for the absolute state). No percent-of-travel is reported — it is `delta / span`, and one obvious name fitted two different quantities. With a camera pose on both frames the rotation is in **world frame** and the translation in **world axes about each camera** — the camera's own *translation* is excluded, since it is in the tracker's gauge and not the object's, so a *translating* camera does show up. Tagged `pose_frame: "oriented" \| "camera"`; exactly one camera pose is an error. `frame_track` is the per-frame series `temporal/derivatives` differentiates | `pose_diff.md` |
| `frames.py` | order frame ids temporally by numeric index (`order_frames`), and resolve a frame window around an `--anchor` by offset (`resolve_window`); reads a run's `layout.json` directly (`--run-dir`) so it needs no hand-typed paths, and `window --emit args` prints the exact `--frames …` args a frame-window consumer needs | [frames.md](frames.md) |
| `mechanism_calls.py` | the mechanism **march** as a checkable artifact: `march` emits `RUN_DIR/mechanism_calls.json` from a pass's mechanism sheets — one call per (joint, swept state), verdicts blank — and `check` is a **gate** (exit code): every articulated joint's states individually called (`expected`/`wrong`/`unsure`), each with its own evidence sentence (pasted lines rejected), stamped against a hash of the joint's declaration so changing `axis`/`origin`/`limit`/`child` stales the march mechanically. Gates finalization and `multiagent.windows plan`; a rigid run owes nothing, and a run whose `modules.json` has the `mechanism` module off (`core/modules.py`) passes trivially | module docstring + [views/mechanism.md](../views/mechanism.md) |

`scorers/silhouette.py`, `check_watertight.py`, and `mechanism_calls.py check`
are **gates** (their exit codes signal pass/fail).
