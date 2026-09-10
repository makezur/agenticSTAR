# `pool` — a resident Blender worker pool (build once, render many)

Today **every render is a cold Blender**: `render.sh` does `exec blender
--background --factory-startup … --python render_wrapper.py`, so every
`shape_pass.sh` pass and every `sweep` pays the full `import bpy` + `build()` + BVH
cost, then exits (see [sweep.md](../views/sweeps/sweep.md): *"each process pays
Blender startup + build(), and EEVEE processes contend for one GPU"*). Scaling to
20–50 keyframes — where poses/joints for different frames are inferred in parallel
— means dozens of cold starts fighting over one GPU.

The pool keeps **G** Blender workers **resident**: each runs `build()` +
`capture_canonical()` **once**, then serves many per-request renders. The order
carries the per-frame pose / joints / camera; the worker holds only the shared
geometry. This is the **G knob** (`--workers`) and the **build-dispatch-to-all-
workers** recycle.

**Two things, two names.** `harness/pool/` is the *code*; the *file queue* it
serves is a run's `spool/` dir (`RUN_DIR/spool` — the same one for a shape pass
and for a windows round's shared pool), always passed as
`--spool`. Renders land under `<spool>/renders/<order-id>/` — never under
`harness/pool/`.

> **Scope.** This is the pool + its file-spool API + a minimal client. The pass
> tool now drives it via **`shape_pass.sh RUN_DIR --pool`** (a throwaway pool per
> pass, spawned with `--once-empty-exit`; see
> [shape_pass.md](../utils/shape_pass.md) § Pool
> rendering) —
> the pool renders the per-frame `match`/`depth`, a cheap cold gauge render still
> produces the turntable + `pose.json` + GLB. That split is about `pose.json` and the
> GLB — shared, once-per-pass artifacts a `--serve` worker never writes — **not**
> about the turntable, which a worker renders perfectly well: `"views": "turntable"`
> is a valid order (serve.py populates the spec/canonical matrices it reads, and
> `setup_lighting` ignores the bbox it is passed, so a resident worker's lighting
> matches a one-shot's). The turntable rides the gauge render because that process is
> already being paid for; moving it into the pool is a speed option, not a
> capability gap. The pool does not persist across passes, and its recycle is keyed
> on the whole-file `scene.py` sha1, so a pose-only edit also recycles.

## Run

```bash
micromamba run -n artscript env PYTHONPATH=harness python -m pool.manager \
    --scene RUN_DIR/scene.py --spool RUN_DIR/spool --workers 2 \
    --tracking CAPTURE/tracking --match-res REF_IMAGE            # match render.sh's render config
# then, from anywhere, drop orders (blocks for the result):
micromamba run -n artscript env PYTHONPATH=harness python -m pool.client \
    --spool RUN_DIR/spool --frame 000040.jpg --views match,depth --id t1
```

`pool.manager` runs until `SIGINT`/`SIGTERM`. Pass `--once-empty-exit` to drain the
current orders and exit (used by batch runs / tests).

### Options

| flag | meaning |
|---|---|
| `--scene` | `RUN_DIR/scene.py`, built once per worker; an edit triggers a recycle |
| `--spool` | spool dir (see layout below) |
| `--workers G` | **G** — number of resident Blender workers (default 2), clamped to `cores-1` AND `16 × GPUs` (see below) |
| `--gpus 0,1` | GPU ids to pin workers to, round-robin. Pinning is `CUDA_VISIBLE_DEVICES` (Cycles) **plus a bwrap `/dev/nvidiaN` device namespace** (EEVEE — which *ignores* `CUDA_VISIBLE_DEVICES`; its EGL context lands on GPU 0 otherwise). Inside an already-bwrapped sandbox nesting is impossible, so the namespace is skipped — there the sandbox launcher's `SANDBOX_GPUS` must have bound the run's GPU subset. |
| `--blender-threads N` | Blender `-t` per worker (default 8; 0 = ncores — avoid: G workers × ncores threads swamps the scheduler for no render gain) |
| `--tracking` / `--match-res` / `--ref-frame` / `--engine` / `--samples` / `--device` | forwarded to each worker so a pool render matches a one-shot `render.sh` with the same flags |
| `--run-dir` | RUN_DIR whose `layout.json` resolves the default-ON `sweep.hand_mask` fill (default: `--scene`'s directory) |
| `--no-hand-default` | disable the `sweep.hand_mask` fill; only orders that carry the key themselves get the hand waiver |
| `--verbose` | forward each worker's Blender stdout/stderr (debugging) |

**The binding
resource is GPUs, not cores or RAM**: `clamp_workers` caps G at
`16 × available GPUs` (`MAX_WORKERS_PER_GPU`, env-tunable via
`POOL_WORKERS_PER_GPU`), where "available" is the `--gpus` list when pinning
or the visible `/dev/nvidia[0-9]*` count otherwise. To use a multi-GPU host,
pass every GPU: e.g. on an 8-GPU host `--workers 128 --gpus 0,1,2,3,4,5,6,7`
round-robins 16 pinned workers onto each card.

## Spool protocol (the file-spool front door)

Deliberately **files, not HTTP** — inspectable, crash-safe, and it does not
re-emulate a per-request `render.sh` fork behind an HTTP server (the exact
cold-start anti-pattern this replaces). Under `--spool`:

```
orders/<id>.json     a client drops a request here (schema below; omit "out")
claimed/<id>.json    the manager MOVES it here (atomic rename) once claimed
renders/<id>/        the manager-allocated output dir the worker renders into
results/<id>.json    the manager writes the worker's result here ATOMICALLY
workers/<slot>/       each worker's own scratch --out (its startup pass dir)
ledger.jsonl         append-only order log (see below)
pool.pgid            the pool process-group id (for the orphan reaper)
```

`<id>` is an immutable output identity within a spool, not a replaceable job
name. `pool/client.py` refuses an id already present in any lifecycle directory;
retries must use a fresh id. This prevents a polling client from accepting an
old `results/<id>.json` and prevents new renders from sharing a directory with
stale images.

**Resetting a spool: archive, don't delete.** Because ids are immutable *within* a
spool, a coordinator starting a new pass needs an EMPTY spool — a pass reuses one
id per frame stem every pass, so a leftover result would be returned as if it were
this pass's render. `locks.archive_spool()` gets that by **renaming** the old spool
to `spool-<timestamp>/` rather than `rmtree`-ing it: the live path is equally empty,
but the previous pass survives for debugging. This matters because a run's
`RUN_DIR/spool` is shared with the pose-refinement **window agents** — their sweep /
`oapply` reports and candidate sheets are the only record of why a pose was accepted.
On by default; see
[shape_pass.md](../utils/shape_pass.md) § The spool is archived, not wiped for the
`--archive-spool` / `--keep-spools` flags. Callers still `reap_pool()` first — the
rename is atomic and won't follow into the tree, so a straggler's writes land in the
archive, but killing orphans is still the primary defence.

**Order** (`orders/<id>.json`) — one JSON object:

```json
{
  "id":     "t1",                     // echoed in the result; also the file stem
  "frame":  "000040.jpg",             // a scene FRAMES key -> resolved EXACTLY as
                                       //   the one-shot path (seeded/authored pose
                                       //   + Pi3X K). The byte-equivalent path.
  "views":  "match,depth",            // which views to run (default "match")
  "joints": {"door_hinge": -45},      // OPTIONAL: override joint states. MERGES onto
                                       //   the frame's states — name only the joints
                                       //   you want changed; the rest keep their
                                       //   committed value. (States are ABSOLUTE: 0 is
                                       //   a POSITION, not an identity, so a replace
                                       //   would straighten every joint you did not
                                       //   name. Honoured by every view, MULTI-FRAME
                                       //   ones included.)
  "sweep":  {"mask":"...", "ranges":"yaw:-15,15,5"},
                                       // OPTIONAL whitelisted per-request sweep
                                       // settings; restored before the next order.
                                       // Scoring masks ("mask", "hand_mask") go
                                       // HERE, never at the top level.
  "seed":   "…/sweep.json#rank=14",   // OPTIONAL: descend from a saved pick —
                                       //   fills the PLACEMENT and joints
                                       //   TOGETHER from any report this repo
                                       //   writes. The placement lands in
                                       //   sweep.pose_start for a pure sweep/apply
                                       //   order (grid centre) and in top-level
                                       //   `pose` for every other view (the
                                       //   frame). NEVER a scale — the engine
                                       //   applies the shared SCALE. See
                                       //   reseed.md; an unresolvable seed
                                       //   FAILS the order.
  "seed_cross_frame": false,           // OPTIONAL: authorize a `seed` whose record
                                       //   belongs to ANOTHER frame (refused by
                                       //   default — usually a copy-paste slip).
                                       //   The pose lands VERBATIM: the camera
                                       //   motion between the frames is NOT
                                       //   compensated, so widen the grid. Logged
                                       //   + recorded in the seed ledger event.
  "visuals": "auto",                  // OPTIONAL panel level: auto | all | none
  "raw":     false,                    // OPTIONAL: bare renders, no panels
  "waive_visual_budget": false         // OPTIONAL: yes, really that many panels
}
```

**Every sweep-family order comes back PANELLED, in two media.** When an order runs
`sweep`, `apply`, `oapply`, `oapply_all`, or `osweep`, the manager builds both from
its renders, `SOURCE | RENDER | OVERLAP | DEPTH` in each, before publishing the
result (`pool/panels.py`). DEPTH is our rendered candidate depth in object units;
it does not load Pi3X:

| artifact | via | answers |
| --- | --- | --- |
| `candidate_sheet_NNN.png` + `candidate_sheet_manifest.json` | `analysis.viz.candidate_sheet` | WHICH of these candidates — several rows per page, IDs indexed in the manifest |
| `side_by_side_<image>.png`, one per candidate | `analysis.viz.sweep_sides` | is THIS candidate right — one candidate across the full width at 512px, where facing and forward-vs-mirrored branding survive |

So a client that sees `results/<id>.json` already has both and never has to pair a
bare render against a source by hand. The
manager can do this because it runs in the analysis env with no `bpy`, so it has cv2 —
the Blender-side worker does not.

Panels are a convenience on top of a render that already succeeded, so a panel
failure is reported in the result and never fails the order. The two halves fail
independently: a strip problem lands in `strips_error` and leaves the sheets, which
are what the manifest indexes, intact. Strips are capped per order
(`pool.panels.MAX_STRIPS`, 12) because a waived `visuals: "all"` order can render
100 candidates and 100 strips is the wrong medium; when the cap bites,
`strips_available` reports the true count and the sheet still paginates every one.

The three panel keys ride at the **top level** (the manager reads them there) and
are also accepted inside the `sweep`/`apply`/`oapply`/`oapply_all`/`osweep` object,
which is
what reaches the engine and so decides how many candidates get rendered at all:

| key | effect |
| --- | --- |
| `"visuals": "auto"` (default) | the winner is always panelled; `dump_topk` adds the best candidate per **ordered grid cell** |
| `"visuals": "all"` | panel EVERY scored candidate, in score order. Refused past 100 panels (`core/visual_budget.py`) unless waived — "all" is a property of the grid, not of your intent |
| `"visuals": "none"` | scores only: no candidate images, no sheet. `<stem>_best.png` is still written, it is the view's own output |
| `"raw": true` | renders, no sheet — the explicit "I'll look at the bare PNGs" escape, at any level |
| `"waive_visual_budget": true` | proceed past the budget |

Multi-frame (object-centric) orders carry their view's own settings object instead of
`"sweep"`, keyed by the view name — `"oapply"`, `"oapply_all"`, `"osweep"`. The SCOPE is
the thing to get right, and it is the view name that says it:

| `"views"` | settings key | scope |
| --- | --- | --- |
| `"oapply"` | `"oapply"` | PER FRAME, named turns — each frame picks its own winner |
| `"osweep"` | `"osweep"` | PER FRAME, searched — each frame runs its own grid |
| `"oapply_all"` | `"oapply_all"` | ONE shared rotation for every frame (a build-orientation fix) |

`"oapply"` (the key, in all three) mirrors `--oapply` including the `flips` /
`flip:x|y|z` candidate-set sugar; `"osweep"` adds `"ranges"` / `"angle_preset"`
mirroring the `--osweep-*` flags; **all three** take `"preset"`
(`--opreset`: an axis plus a range of angles, e.g. `"preset": "z:quarters"` — the
one spelling shared by every object-centric view; precedence
`ranges` > `preset` > `angle_preset`, and a `preset` given alongside a candidate set
scores their UNION), and all three take an optional
`"frames": "a.jpg,b.jpg"` to scope the view to a SUBSET of the scene's frames (in
shared `oapply_all` the report then switches to subset mode — see oapply_all.md).
There is no `"osweep_all"`: a shared rotation ranked by MEAN IoU can be mediocre in
every frame, so the whole-run path is imperative only and the name fails loudly with
that message.

The DOF grammars an order's `ranges`/`oapply` strings speak are defined by the
view docs, not here: camera-frame `roll/yaw/pitch/dpx/dpy/tz` + absolute
joint states in [sweep.md](../views/sweeps/sweep.md), object-canonical
`rx/ry/rz` + the `flips` panel in [osweep.md](../views/sweeps/osweep.md) /
[oapply.md](../views/sweeps/oapply.md) / [oapply_all.md](../views/sweeps/oapply_all.md) — two
DIFFERENT frames; each parser redirects
you if you use the other's spellings. Their `masks_dir` /
`hand_masks_dir` are filled from the run's `layout.json` when the order doesn't
carry the key, same default-ON contract as `sweep.hand_mask` below.
(Advanced: an explicit `"pose": {...}` renders a pose not in the scene; its K
falls back to the placeholder — a scene `frame` is the tested path.) **Do not set
`"out"`** — the manager allocates `renders/<id>/` as the SOLE allocator, which is
what removes the `iterations/NNNNNN/renders/NNNN` check-then-create race that a
pool would otherwise trip. Unknown top-level keys are **rejected** (`ok:false`), so a
misplaced scoring key (e.g. `hand_mask` beside `sweep` instead of inside it) cannot
be dropped silently and downgrade the sweep gate from `iou_visible` to `iou_raw`.

**Hand-occlusion waiver is default-ON.** At claim time the manager fills
`sweep.hand_mask` with `<hand_masks_dir>/<frame stem>.png` from the run's
`layout.json` whenever the order's `sweep` object doesn't carry the key and the
per-frame file exists — so hand-held sweeps gate on `iou_visible` without every
client restating the path. An order that *does* carry the key has decided:
`"hand_mask": ""` is the explicit per-order opt-out. `--no-hand-default`
disables the fill pool-wide; `--run-dir` points the fill at a different
layout (default: `--scene`'s directory).

**The scoring mask is filled too.** Same join, same layout (`masks_dir`), same
"an order that carries the key has decided" contract — but `sweep.mask` is
**required**, not a waiver: without it `engine._prepare` prints `--sweep-mask … is
required; skipping` and returns *before the first render*, which the worker would
report as `ok: true` with an empty render dir. So the mask is derived here, and a
sweep-family view that runs without writing its report fails the order
(`_missing_reports`), so *any* skip path — mask, joints, empty grid — is loud.

**`seed` — the next hop of a coordinate descent.** Fitting a frame is a chain of
sweeps, each starting from a candidate picked by eye in the previous one; the pose
and the articulation of that pick live in **two different order keys**
(`sweep.pose_start` and top-level `joints`), and `pose_start` is placement-only, so
hand-carrying the pair is where a hop gets quietly half-dropped — the joint reverts
to the frame's committed state while the pose carries, and the render still looks
plausible. `"seed": "<path>#<selector>"` names the pick instead and the manager
fills **both** at claim time, from any pose-bearing JSON this repo writes (a sweep
report by `#rank=N` / `#candidate_id=X` / `#best`, an object report or a
`progress.json` / `poses.json` / fragment by `#<frame>.jpg`, a `mesh/pose.json`, or
another order). An explicit `sweep.pose_start` / `pose` / `joints` beside it **wins**, as with the
mask fills; **unlike** them an unresolvable seed FAILS the order rather than
degrading, because the degraded render is a believable picture of the wrong
configuration. The key is consumed — the claimed order carries `seed_from` instead —
and a `seed` ledger event records the descent. Every `sweep.txt` ends with a
paste-ready NEXT HOP order in this shape. Full contract: [reseed.md](reseed.md).

**Result** (`results/<id>.json`): `{ "ok": true, "out": ".../renders/t1",
"views": ["match","depth"], "seconds": 0.9 }` on success, or `{ "ok": false,
"error": "..." }` on any per-request failure (a bad request never takes the
resident worker down). Written temp-then-`os.replace`, so a polling client never
reads a half-written file (the harness has no other atomic-write helper).

A panelled order also carries `"panels"`: `{"pages":
["candidate_sheet_001_preview.png", ...], "original_pages":
["candidate_sheet_001.png", ...], "manifest": "candidate_sheet_manifest.json",
"strips": ["side_by_side_sweep_top_00_preview.png", ...], "original_strips":
["side_by_side_sweep_top_00.png", ...]}` (paths relative to `out`). When a
preview cap does not reduce an image, its default and original path are identical.
The block may instead be
`{"skipped": "raw" | "visuals=none" | "not-panelled" | "<why>"}`, or
`{"error": "..."}` when the sheet could not be built. `ok` stays `true` either
way. Two optional keys ride with the strips: `"strips_available": N` when more
candidates were rendered than got a strip (the cap), and `"strips_error"` when the
strips failed but the sheets did not — in that case `pages` is still there and `ok`
is still `true`.

Set `pool.candidate_sheet_max_dimension` and
`pool.side_by_side_max_dimension` in `run_config.json` to target the longest edge
of default preview pages and strips (`0` selects the native file directly).
Previews never downsample below `0.5x` native linear resolution. The canonical
`candidate_sheet_NNN.png` and `side_by_side_*.png` files always retain the native
completed panel; reduced derivatives use `_preview.png`. Manifests/results record
both paths, dimensions, and scale so an agent can start cheaply and escalate to
the original without rebuilding.

## `ledger.jsonl` — the order log (for debugging)

`claimed/` and `results/` already say WHAT was ordered and how it turned out, but
not *when*, *where*, or *how many tries*: order timing, the worker slot that ran
it, and worker deaths that a retry papered over exist only in the manager's
`log()` narration — which goes to **stdout**, and `pool/session.py` spawns the
manager with stdout inherited, so for a `shape_pass` / `multiagent.pool_session` pool that
narration lands wherever the coordinator's stdout went and is gone once the pool
exits. `<spool>/ledger.jsonl` persists it beside the spool it describes:

| event | written by | carries |
|---|---|---|
| `submit` | `client.submit` (the client PROCESS) | `id`, `frame`, `views`, `pid` |
| `claim` | manager intake | `id`, `frame`, `views`, `queued` (queue depth) |
| `seed` | manager intake, when an order carried `seed` | `id`, `seed_from` (the resolved address), `filled` (which of pose_start/joints came from it), `joints` (the seeded state) |
| `result` | manager slot, **after** `results/<id>.json` | `id`, `ok`, `worker`, `seconds`, `error`, `panels` |
| `worker_died` | manager retry path | `id`, `worker`, `attempt`, `before_render` |
| `pool_ready` / `recycle` / `pool_stop` | manager lifecycle | `workers`, `scene_sha1`, `gpus`, `pid`/`pgid`, `pending` |

`submit` is logged by the **client** because the manager only ever sees an order
at claim time — the submit→claim queue wait is measurable nowhere else. `result`
is logged *after* the result file lands, so the ledger is always data trailing a
delivered order, never a gate on delivery. Every append is **best-effort**: it is
one `os.write` on an `O_APPEND` fd (so G slot threads and separate client
processes cannot tear each other's lines) and swallows every `OSError` — a render
that succeeded must never become `ok:false` because a log line didn't land.

The ledger sits at the spool ROOT, so `archive_spool`'s rename carries a pass's
whole history aside with it and a fresh pass starts a fresh ledger.

Read it back:

```bash
micromamba run -n artscript env PYTHONPATH=harness python -m pool.ledger \
    --spool RUN_DIR/spool              # per-order timeline + summary
    #        ... --failures            # only orders that failed or never finished
    #        ... --raw                 # the events, one JSON per line
```

The report's `ran` is the slot's wall-clock claim→result; `render` is the worker's
own reported seconds. `render` << `ran` means the time went elsewhere in the slot
(panels, a retry, a respawn), and `queued` >> 0 means the order waited on a busy
pool rather than on Blender — the first two questions a slow pass raises.

## Recycle — "build dispatch to all workers"

The manager watches `--scene`'s **sha1** (the same whole-file hash
`render_wrapper` stamps into `MANIFEST.json`). On a change it **drains** in-flight
requests, **`SIGTERM`s all G workers**, and **relaunches G fresh** ones — each
re-runs `build()` on the new geometry. Clean `--factory-startup` `bpy` each time,
identical to today's proven cold path, just amortized across the many requests
between edits.

Because the recycle triggers on the whole-file hash, a **pose-only** edit to
`scene.py` (changing `FRAMES`) also recycles even though the geometry is
unchanged. Isolating a `build()`-only geometry fingerprint to skip that is a
deliberate later refinement — the order already carries the pose, so most
per-frame pose exploration needs **no** scene edit at all.

## Transport (worker ↔ manager)

Each worker is a `blender --serve` child; the manager talks to it over **pipes**:
requests on the worker's stdin (one JSON line each), replies on its stdout. The
render views print `[render_wrapper]` diagnostics to stdout, so every control
message is a single line prefixed with `@@POOL@@ `; the manager scans for that
prefix and treats every other line as a forwarded log. A worker emits one
`{"event":"ready"}` after `build()` completes, then one `{"event":"result"}` per
request.

## Verification

- **Byte-equivalence** (the correctness bar): a `--serve` worker's
  `match_<stem>.png` / `depth_<stem>.npy` for a frame equals a one-shot
  `render.sh` render of the same frame/scene — the pool changes only wall-clock,
  never pixels. (Use a per-frame view set, no `turntable`: lighting is set up once
  at serve start, matching a single-frame one-shot exactly.)
- **Smoke / warmth / recycle**: `pool.manager --workers 2` on a run dir; N orders
  via `pool/client.py` land N results, each in its own `renders/<id>/`, with
  `RUN_DIR/mesh/pose.json` untouched; orders 2..N skip `build()`; a `scene.py`
  edit recycles all workers before the next order.
