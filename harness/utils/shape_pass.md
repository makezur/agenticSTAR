# `shape_pass.sh` — render and score the ONE committed state, on every frame

This is the main beat of the **shape loop**: author `RUN_DIR/scene.py`, render +
score every frame, look at the images, repeat. It produces the full diagnostic
set. After a pose-windows apply, use `composite_pass.sh` for the routine match
renders and composites; use this full pass only when its additional shape,
depth, mechanism, aggregate, or self-intersection outputs are needed.

A pass is a fixed sequence that is otherwise run by hand: `render.sh`
(`match,turntable,depth`) → then, **per frame**, a `composite.py` call that threads
the **same** pass dir through **six** path arguments plus a `depth.py` call → then
`aggregate.py`. `shape_pass.sh` runs that whole sequence from one command, so:

- the **pass dir is captured** from what `render.sh` prints (`PASS <name> OUTPUT
  DIR: …`) and reused everywhere — the pass number is **never hand-typed** into six
  paths;
- the per-frame paths (`match_<stem>.png`, `metrics_<stem>.json`, …) are built once
  from the frame stem;
- it writes **`depth_<stem>.json`** per frame — the file `aggregate.py` reads for
  its depth roll-up (the manual step-4 `composite.py` command alone never wrote it);
- it always writes a **self-intersection report** for the state it rendered (below).

It does not add any new analysis — it drives the existing tools exactly as their
own docs specify. Read the individual images yourself every pass; this only
removes the boilerplate of producing them.

## Run

```bash
harness/utils/shape_pass.sh RUN_DIR --frames 000000.jpg,000040.jpg
harness/utils/shape_pass.sh RUN_DIR                     # frames from layout.json
harness/utils/shape_pass.sh RUN_DIR --pass-label coarse-shape
harness/utils/shape_pass.sh RUN_DIR --pool --workers 3  # render via the resident pool
```

`RUN_DIR/scene.py` must exist. Each call opens a global shape iteration, or can
attach to an open pose iteration for a deliberate full checkpoint. Renders go to
`RUN_DIR/iterations/NNNNNN/renders/NNNN/`; the harness records the pose and
scene and commits completed work automatically. A process-lifetime lock at
`RUN_DIR/.pool.lock` rejects a second render coordinator for the same run.

Full Blender stdout/stderr is written to
`RUN_DIR/logs/render-<timestamp>-<pid>-<suffix>.log` instead of being replayed
through the caller's terminal. `shape_pass.sh` prints the log path and final
`PASS_DIR`; on render failure it also prints the last 50 log lines.

### Options

| flag | meaning |
|---|---|
| `--frames a.jpg,b.jpg` | frames to render + score (default: the `frames` in `layout.json`) |
| `--existing-pass DIR` | recover an open iteration by scoring an existing pass under its `renders/` directory, without rerendering |
| `--bg-mode alpha\|black` | backdrop for the composite panels (default `black`; the scaffolded config pins `alpha`). Config: `visuals.bg_mode` |
| `--pool-timeout S` | per-order wait timeout for a pool render (default 600) |
| `--engine CYCLES` | render engine (default `BLENDER_EEVEE_NEXT`; use `CYCLES` for a final pass) |
| `--samples N` | render samples (default 64) |
| `--turntable-views ring4\|sphere8` | turntable view set per articulation state: `sphere8` (default — a level ring plus a raised and a dropped pair, so a caved-in top or unbuilt base is visible) or `ring4` (the level ring only, half the renders per state). Turntable cost is **states × views**. See [../views/turntable.md](../views/turntable.md) |
| `--turntable-jitter SEED` | perturb the turntable directions for a deliberate look-from-somewhere-new pass. Unset (default) keeps every pass on the **same** directions, which is what makes one iteration's views comparable with the last's |
| `--pass-label NAME` | store optional manifest metadata; directories remain numeric and `final` is rejected |
| `--no-aggregate` | skip the closing `aggregate.py` roll-up |
| `--depth-report` / `--no-depth-report` | score + panel Pi3X depth this pass (`depth_<stem>.json`, the depth-residual images, the `depth_mae_canon` summary column). `--no-depth-report` skips depth **scoring only**; every downstream depth reporter then goes quiet on its own, but the depth view is still rendered (see `--no-depth-render`). Camera seeds / `measure_depth` are unaffected. Default on. Config: `depth_config.json` `report` |
| `--depth-render` / `--no-depth-render` | render the depth view this pass (`depth_<stem>.npy`), which feeds the depth scorer **and** the GT-free object-units `DEPTH` panel / [depth sheet](../analysis/viz/depth_units.md). `--no-depth-render` drops the view — one render per frame per pass cheaper, and every depth visual then has nothing to draw. Independent of `--depth-report`; this is the switch for a run that should not pay for depth at all. Default on. Config: `depth_config.json` `render` |
| `--ncpu N` | **N** — CPU scoring workers, concurrent `composite`+`depth` subprocesses (default `min(4, cpu_count)`; `1` = serial). Config: `concurrency.ncpu` |
| `--pool` / `--no-pool` | render per-frame `match`/`depth` through the resident **render pool** instead of a cold `render.sh` (see below). Default off. Config: `pool.enable` |
| `--workers G` | **G** — resident Blender workers when `--pool` (default `2`). Config: `concurrency.workers` |
| `--gpus 0,1` | GPU ids the pool pins workers to, round-robin (default: none — Blender sees all). Config: `pool.gpus` |

All frames are scored numerically in parallel with `--ncpu` CPU workers. This
changes wall-clock only; subprocess outputs are byte-identical to serial scoring.
A hard failure in any CPU scorer aborts the whole pass.

## Config — `RUN_DIR/run_config.json` (optional)

The concurrency knobs **G / N** and the visual presentation settings can live in a
per-run `run_config.json` beside `layout.json` / `depth_config.json`, so they need
not be re-typed on every call. Precedence is the same as `depth_config.json`:
**explicit CLI flag > `run_config.json` > built-in default**. Every field is
optional; an **absent (or corrupt) file falls back to the CLI defaults**, so older runs
are unchanged. The legacy `iterate_config.json` is still read when no
`run_config.json` is present.
`run.sh` scaffolds one for every new multi-frame run (via
`python -m utils._run_config`, whose `scaffold_config` is the full-size
profile: pool on, 16 workers/ncpu per GPU (`MAX_WORKERS_PER_GPU`),
`bg_mode` "alpha"; `--workers/--ncpu/--gpus` on run.sh
override the sizes). A re-run never clobbers an existing file. Delete it to
fall back to pure CLI defaults, or edit it to pin different values for the run.
Full schema (values shown are the built-in CLI defaults, NOT the scaffold's):

```json
{
  "concurrency": { "workers": 3, "ncpu": 4 },
  "visuals": { "bg_mode": "black" },
  "pool":  { "enable": false, "gpus": [0],
             "candidate_sheet_max_dimension": 0,
             "side_by_side_max_dimension": 0 }
}
```

`pool.candidate_sheet_max_dimension` and `pool.side_by_side_max_dimension`
control derived preview pages and strips. `0` selects the native image directly.
A positive value writes an `_preview.png` derivative, preserving aspect ratio and
never shrinking below `0.5x` native linear resolution. Canonical
`candidate_sheet_NNN.png` and `side_by_side_*.png` files stay native. The sheet
manifest, pool result, and side-by-side `.variants.json` sidecars record both paths,
sizes, and scale for preview-first inspection followed by exact-raster escalation.

### Depth switches — `RUN_DIR/depth_config.json`

The two depth flip switches live in `depth_config.json` (**not**
`run_config.json`) — the same per-run file that already carries the depth
`backend` / `conf_thr`, so the depth backend and its on/off sit together, and both
the analysis scorers and the Blender `sweep` resolve them by the same walk-up.
`run.sh` fills `backend` from the capture manifest's `depth` kind (`pi3x`; a
`none` capture scaffolds both switches off). Both default **on**
(an absent field / missing file behaves exactly as before), are **independent**,
and neither touches the per-frame camera seeds or `measure_depth` — those read the
pointmap directly and always work:

```json
{ "backend": "pi3x", "conf_thr": 0.1, "cost": true, "report": true }
```

- **`cost`** — depth **supervision** in the `sweep` ranking
  (`combined = IoU − weight · depth_canon`). `false` forces `depth_weight 0` (pure
  IoU), regardless of `--sweep-depth-weight`.
- **`report`** — the depth **scorer / panels** (`depth_<stem>.json`, the
  depth-residual images). `false` makes
  `shape_pass` skip depth **scoring**; every downstream reporter (summary,
  `aggregate`) reads those files, so it goes quiet on depth
  automatically. It does **not** stop the depth render — see `render`.
  `shape_pass`'s `--depth-report`/`--no-depth-report` overrides this per pass.
- **`render`** — the depth **view** itself (`depth_<stem>.npy`). Independent of
  `report`, because our own depth render needs no GT: it feeds the object-units
  `DEPTH` panel and the [depth sheet](../analysis/viz/depth_units.md) even when
  nothing is being scored, which is what makes those visuals work on a monocular
  run. `false` drops the view from the render (and from pool orders), saving one
  render per frame per pass. `shape_pass`'s `--depth-render`/`--no-depth-render`
  overrides this per pass.

`run.sh` scaffolds all three to `true`; pass `--no-depth-cost` /
`--no-depth-report` / `--no-depth-render` to scaffold them off. A capture with no
depth (`depth: none`) forces `cost` and `report` off automatically but LEAVES
`render` on — the monocular case is exactly where the GT-free depth visuals earn
their keep. Set all three `false` to run with depth completely disabled while
still using the capture's cameras.

## Pool rendering — `--pool` (resident Blender, build once, render many)

By default each pass **cold-starts** Blender once via `render.sh` (build + BVH, then
render every frame, then exit). With `--pool`, `shape_pass` instead uses the resident
**render pool** ([pool/README.md](../pool/README.md)) — **G** workers that each
`build()` once and then serve many renders — for the expensive **per-frame
`match`/`depth`** fan-out.
Because a serve worker only ever writes into its own manager-allocated output dir and
never the shared `pose.json`/GLB, `--pool` runs a **hybrid**:

1. **Gauge render** — one cheap cold `render.sh --views turntable --export`, producing the
   per-**state** `turntable_*_az*_el*.png`, the pass-local `pose.json` (also copied to `mesh/pose.json` as the latest artifact), the GLB,
   and the pass dir. Turntable is per-state (not per-frame), so this stays cheap. It
   rides this render because the process is already being paid for, **not** because a
   worker couldn't do it — `"views": "turntable"` is a valid pool order.
2. **Pool** — a throwaway `pool.manager --once-empty-exit` over `RUN_DIR/spool`; one
   `match,depth` order per frame; the G workers drain them in parallel. Each pass must
   START from an empty spool (order ids are frame stems, reused every pass, so a
   leftover `results/<id>.json` would be mistaken for this pass's render). By default
   the old spool is **archived, not deleted** — see below.
3. **Reconcile** — each order's `match_<stem>.png` / `depth_<stem>.npy` is copied into the
   gauge pass dir, so the pass dir ends up **identical** to the cold path and every
   downstream step (scoring, `read_scale`, the summary) is unchanged.

The pool changes only wall-clock, never pixels — a pooled `match_<stem>.png` /
`depth_<stem>.npy` is byte-equivalent to the cold render. Any failed order aborts the
pass (same as `render.sh` returning non-zero). Leave `--pool` off (the default) for the
plain single-cold-render path.

### The spool is archived, not wiped (`--archive-spool`, default ON)

`RUN_DIR/spool` is **not** this tool's private scratch. A pose-refinement window agent
submits its sweep / `oapply` orders to the SAME spool, and `renders/<order-id>/` holds
the `sweep.json` reports and `candidate_sheet_*.png` pages that are the only record of
*why* a pose was accepted.

Default: the previous spool is **renamed** to `RUN_DIR/spool-<YYYYmmdd-HHMMSS>/`.
The live path ends up just as empty, so the hermetic-pass guarantee is unchanged, but
every order, report, and panel stays readable.

| flag | meaning |
|---|---|
| `--archive-spool` / `--no-archive-spool` | rename aside vs the old delete (default: archive). Config: `pool.archive_spool` |
| `--keep-spools N` | retain the newest N archives, pruning oldest first (default 8; `0` = keep every archive forever). Config: `pool.keep_spools` |

Cost is small (a drained spool is on the order of 10 MB). The
rename is atomic and does not follow into the tree: a straggler worker's open fds
follow the inode into the archive and can never land in the fresh spool. `reap_pool`
still runs FIRST; the archive is a safety net, not a substitute for killing orphans.

The other pool coordinator, `multiagent/pool_session.py`, has no equivalent flags
on purpose: a windows round's pool never clears the spool at all. It reaps orphaned
PROCESSES only and leaves every cached render in place, because the refiners
submitting into that spool treat completed work as a cache across a re-run.

## Layout comes from `RUN_DIR/layout.json`

`run.sh` writes `RUN_DIR/layout.json` recording the resolved layout (distinct from
the per-pass `MANIFEST.json` in `iterations/NNNNNN/renders/NNNN/`).
All paths point into the run's CAPTURE (the canonical materialized sequence dir —
see [datasets/README.md](../../datasets/README.md)):

```json
{
  "kind": "multiview",
  "capture": "/abs/.../captures/<name>",
  "frames_dir": "/abs/.../captures/<name>/frames",
  "masks_dir": "/abs/.../captures/<name>/mask_object",
  "hand_masks_dir": "/abs/.../captures/<name>/mask_hand",
  "ref_frame": "000000.jpg",
  "frames": ["000000.jpg", "000040.jpg"]
}
```

`shape_pass.sh` reads this so it never re-derives where the images/masks/tracking
live; the tracking dir (cameras) is `<capture>/tracking`, derived from the
`capture` entry. **There is no convention-based path guessing** — guessing would
silently read the wrong files. To override a field, pass the layout on the CLI —
these take precedence over the layout file:

```
--tracking DIR  --frames-dir DIR  --masks-dir DIR  --hand-masks-dir DIR  --ref-frame f.jpg
```

The per-frame source (`frames_dir/<frame>`) and object mask (`masks_dir/<stem>.png`)
must exist or `shape_pass.sh` stops early with a clear message. The hand mask
(`hand_masks_dir/<stem>.png`) is used **only if that frame has one** — frames
without it just don't get `--hand-mask`.

## What each frame produces (in the pass dir)

Each pass also contains its authoritative `pose.json`, `MANIFEST.json`, and
`scene_snapshot.py` (the scene it rendered — see Lineage below). Per-frame outputs are `match_<stem>.png`, `overlap_<stem>.png`, `metrics_<stem>.json`,
`depth_<stem>.npy`, `depth_<stem>.json`, `depth_residual_<stem>.png`,
`composite_<stem>.png`, and `side_by_side_<stem>.png` for every frame. The focused
side-by-side image contains labeled source, masked source, and match panels; the
masked source and match use the configured `visuals.bg_mode` backdrop. After
completion, bookkeeping publishes
the per-frame composites as
`RUN_DIR/iterations/<iteration>/composites/<stem>.png`; open or failed
iterations retain their pass-local outputs but do not receive this promoted
set. The run-level `report.json` / `report.txt` come
from `aggregate.py`. Every pass also writes
`self_intersection.txt` / `.json` — see below.

## Lineage

`MANIFEST.json` carries the pass-local `scene_snapshot.py` and `scene_sha1`.
When the global iteration completes, bookkeeping also preserves `scene.py` and
`pose.json` at the iteration root and commits them in the run's private Git
repository. Compare completed iterations with:

```bash
git -C RUN_DIR log --oneline
git -C RUN_DIR diff <older-commit> <newer-commit> -- scene.py mesh/pose.json
```

The snapshot and `scene_sha1` are both taken from **one read of `scene.py`, made
before the scene is exec'd** — so they describe the bytes that actually produced
these renders. That ordering is the whole guarantee: renders take minutes, and a
fingerprint taken at the *end* of a pass would record an edit that landed
mid-render as the scene that rendered, which it never was. (The run lock
serializes coordinators, not your editor.)

Because the two values come from one read, hashing a snapshot back MUST reproduce
its manifest's `scene_sha1` — a mismatch means the pass dir was tampered with or
truncated, and the lineage should not be trusted.

`harness_rev` records the harness revision. `scene.py` imports the harness
(`from shapes import box`), so a snapshot is a faithful record of the authored
delta — what a diff needs — but is **not** replayable on its own: re-execing it
needs the harness at that revision. A `-dirty` suffix means the imports cannot be
pinned to anything at all.

See [BOOKKEEPING.md](../BOOKKEEPING.md) for allocation, recovery, and portable
history bundles.

### `--intent` / `--rollback` (write down why, before rendering)

`--intent "what this pass changes"` and `--rollback "what to keep if it is
rejected"` are recorded to `<pass>/intent.json`. Both are stated on the command
line, so they are fixed **before** any score exists to argue from — and the
file is written by the render itself **at pass allocation**, before anything
renders, so a run that dies mid-render still leaves the declaration behind
(the interruption it exists to survive).

`--rollback` earns its keep on a pass that changes two independent things. A
geometry edit plus one frame's pose reads, to whoever resumes the run, as a
single atomic "experiment" — so rejecting the bad half discards the good half
with it, and the reason the pose was worth keeping died with the context that
held it. Write the scope down and it survives:

```bash
harness/utils/shape_pass.sh RUN_DIR --pass-label fore-edge-and-hinge \
    --intent "fore-edge solid + cover extents; 000250 hinge 130->146.8" \
    --rollback "revert the geometry, KEEP the 000250 hinge"
```

Omit both flags and no file is written — this records a declaration, and
inventing one would defeat the point of making it.

The closing summary prints each frame's `iou_raw` / `iou_visible` /
`depth_mae_canon` read back from the JSON, so you see the gates without opening files.

## Self-intersection (every pass, mandatory)

Every pass runs the
[self-intersection report](../analysis/self_intersection.md) on the state it just
rendered — no flag skips it. It answers whether that configuration is physically
possible (two watertight parts cannot share volume), from the pass's own
`pose.json` + the exported GLB: no renders, no network, ~1 s.

The table is `PASS_DIR/self_intersection.txt` (+ `.json`); the summary prints the
deepest part-pair overlap, the **onsets** (the step *into* that frame drove the
parts together — the row to read in the temporal report), and any part excluded as
non-watertight (its overlaps are **unknown, not zero**).

Still a **warning, not a gate**: no threshold decides anything, and a small
modelled contact is fine once you can name it. A non-zero exit means the report
could not be *produced* (unreadable GLB, incomplete joint states), which fails the
pass like any other hard scorer. The read is the orchestrator's, never a
pose-refiner's.

Then **read the individual images** (`side_by_side_`, `overlap_`,
`depth_residual_`) plus **one turntable sheet per articulation state**
(`turntable_sheets/turntable_sheet_<state>.png` — every orbit view of that state on
one page, each tile labelled with the direction it was shot from) as the iteration
protocol requires. `shape_pass.sh` produces the images and surfaces the numbers;
it does not replace looking at them.

## Running the sequence by hand (the raw commands)

`shape_pass.sh` is the normal path. Run the raw commands only for a one-off (a
single `sweep`, or a single-image run `shape_pass.sh` doesn't drive). The sequence
per pass is `render.sh` → per frame `composite.py` **and** `depth.py` →
`self_intersection.py` → `aggregate.py`. `PASS_DIR` is the
output directory `render.sh` printed; `PI3X` is the capture's tracking dir
(`CAPTURE/tracking`; the capture root also works for the analysis tools).

```bash
# 1. render (prints PASS_DIR):
harness/render.sh RUN_DIR/scene.py RUN_DIR/views \
    --views match,turntable,depth --tracking CAPTURE/tracking --frames <all> \
    --ref-frame <ref> --match-res <ref image> --export RUN_DIR/mesh/object.glb

# 2. per frame — composite (metrics + overlap + depth-residual panels):
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.viz.composite \
    --source <frame image> --render PASS_DIR/match_<frame>.png \
    --mask <frame mask> [--hand-mask <hand mask>] \
    --metrics-out PASS_DIR/metrics_<frame>.json \
    --overlap-out PASS_DIR/overlap_<frame>.png \
    [--tracking CAPTURE/tracking --render-depth PASS_DIR/depth_<frame>.npy \
     --depth-residual-out PASS_DIR/depth_residual_<frame>.png] \
    --side-by-side-out PASS_DIR/side_by_side_<frame>.png \
    --out PASS_DIR/composite_<frame>.png

# 3. per frame — depth.py writes depth_<frame>.json (the file aggregate.py reads;
#    composite.py above does NOT write it):
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.scorers.depth \
    --tracking CAPTURE/tracking --render-depth PASS_DIR/depth_<frame>.npy \
    --mask <frame mask> --source <frame image> --frame-name <frame> \
    [--hand-mask <hand mask>] --out PASS_DIR/depth_<frame>.json

# 4. the turntable sheets — one page per articulation state (shape_pass.sh does this
#    for you; run it by hand for an older pass):
micromamba run -n artscript env PYTHONPATH=harness \
    python -m analysis.viz.turntable_sheet --pass-dir PASS_DIR

# 5. roll-up:
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.rollup.aggregate \
    --run-dir RUN_DIR --views-dir PASS_DIR
```
The turntable glob is `turntable_*_az*_el*.png` (files are keyed by articulation
**state**, not frame, and carry the azimuth/elevation they were shot from — see
[../views/turntable.md](../views/turntable.md)).
