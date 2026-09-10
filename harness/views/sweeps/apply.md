# `apply` view — "tilt 90°. Boom. Render. What's the IoU?"

**Answers:** "apply THIS directional order — tilt/turn/orbit/open — to the frame's
current pose, show me the render, and tell me how well it scores."

`apply` is the **imperative / directional** sibling of [`sweep`](sweep.md).
Where `sweep` **searches** (score a whole grid of candidates against a mask, rank,
pick the best), `apply` scores **exactly the one config you name**: it composes the
order onto the frame's current pose, renders it at full match-res, scores that single
silhouette against the mask (IoU, + a Pi3X depth penalty with `--tracking`), and writes a
**paste-ready** pose/joints block. You read the IoU / depth-loss verdict and the
side-by-side and decide "yep, that's it" — no search.

Under the hood `apply` **is** a `sweep` of one candidate: it runs the same scoring
engine, so the report, the `iou_raw`/`iou_visible`/`depth_canon`/`combined` numbers,
the depth-loss verdict, and the paste block are identical in form to `sweep`'s — just
named `apply.*`. Use `apply` when you already know the move you want to *check*; use
`sweep` when you want the tool to *find* the move that best fits the photo.

## The order — a unified DOF vector

`apply` takes **one absolute value per DOF** (contrast `--sweep-ranges`'
`min,max,steps`). Mix pose orders and joints freely (see
[`../../rig/lie.py`](../../rig/lie.py) `apply_order`):

- **pose "order" DOFs** — camera-frame increments composed onto the frame's pose,
  about the object centre:

  | DOF | meaning |
  |---|---|
  | `roll` | in-plane rotation about the optical axis (+Z), deg |
  | `yaw` | out-of-plane orbit about camera-up (−Y), deg |
  | `pitch` | out-of-plane orbit about camera-right (+X), deg |
  | `dpx` | image-plane shift RIGHT, a **fraction of frame width** |
  | `dpy` | image-plane shift DOWN, a **fraction of frame height** |
  | `tz` | depth nudge along +Z, a **fraction of the object's depth** |

  You type a SIGNED value per DOF, so the sign convention matters: what a
  positive `roll`/`yaw`/`pitch` (and every other DOF) does on screen is the
  verified table in
  [`../../../conventions/DIRECTIONS.md`](../../../conventions/DIRECTIONS.md) — read it
  before you pick a sign (roll + = clockwise, yaw + = left edge closer, pitch +
  = top edge closer).

- **joint DOFs** — the **absolute** state (revolute: degrees; prismatic: canonical
  units) of any joint your `scene.py` declares in `JOINTS`, clamped to the joint's
  `limit`. A pose order is a *delta* on the base pose; a joint value is the *absolute*
  state — the same split as `--sweep-ranges`.

## Run
```bash
# tilt the reference frame 90° about camera-up, render + score against the mask:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views apply \
    --match-res IMAGE --sweep-mask MASK --apply 'yaw:90'

# a visually directed pitch + a hinge opening on a specific frame, depth-aware:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views apply \
    --frame 000040.jpg --match-res IMAGE --sweep-mask MASK --tracking CAPTURE/tracking \
    --apply 'pitch:17;dpx:0.05;door_hinge:70'

# apply AND render match in one launch — match shows the ORIGINAL pose (apply is
# non-destructive), apply shows + scores the moved one:
harness/render.sh RUN_DIR/scene.py RUN_DIR/views --views match,apply \
    --match-res IMAGE --sweep-mask MASK --apply 'yaw:90'
```

`--apply` (required) — the order, `';'`-separated `dof:value` (ONE value each). `dof`
is a pose order (`roll,yaw,pitch,dpx,dpy,tz`) OR a declared joint name; mix
freely. Runs against the `--frame` frame (default: the reference).

`apply` **reuses `sweep`'s scoring flags** (same interface, no parallel set): every
flag's `--help` (see [`../../rig/args.py`](../../rig/args.py)) is the ground truth.
- `--sweep-mask` (required) — the object mask the applied silhouette is scored against.
- `--sweep-hand-mask` / `--sweep-hand-dilate` — waive an occluder; `iou_visible` gates.
  `--sweep-hand-dilate` defaults to **0** (the hand mask exactly).
- `--sweep-depth-weight` (with `--tracking`) — Pi3X depth penalty weight; `combined =
  gate IoU − weight × min(depth_canon, 1)`, identical to `sweep`.
- `--sweep-quality` — scoring-render resolution (the winner is re-rendered at full
  match-res regardless).
- `--sweep-pose-start` — JSON pose to apply the order onto (default: the frame's pose).

## Outputs (in the pass dir `render.sh` prints)
- `apply_best.png` — the applied configuration re-rendered at **full match res**.
- `apply.txt` — a sweep-style legend, the `iou_raw / iou_vis / dp_canon / combined`
  line, the `depth loss` verdict (with `--tracking`), and the **paste-ready** `"pose"`
  block (iff pose DOFs) and/or `"joints"` block (iff joint DOFs). Same format as
  `sweep.txt`'s BEST block. There is no `SCALE` line: the order is rigid, so the
  applied config always carries the scene's `SCALE` unchanged.
- `apply.json` — machine form: `view:"apply"`, `best` (pose/joints + IoUs +
  `depth_canon`/`combined` + `apply_best.png`), `pose_start`, `ranges`, `passes`,
  `status`. Mirrors `sweep.json` (a single-candidate `ranked`). `passes`
  is one entry — `apply` is a one-point grid and never refines — but the field is
  present so a reader parses both verbs the same way.

## The IoU here is a fast PROXY
Exactly as in `sweep`: the per-config IoU is a low-res numpy proxy (~0.005 of the
authoritative score). Re-verify a config you keep — paste `best` into the frame,
render `match,depth`, and run [`silhouette.py`](../../analysis/scorers/silhouette.md).
With `--tracking`, ranking/scoring is depth-aware; don't keep a config whose `depth loss`
reads `HIGH`.

## Non-destructive — like every view
`apply` restores the base pose + render resolution/intrinsics (the sweep engine does
this), so a co-requested `match`/`turntable`/`depth` renders the **original** pose,
and `pose.json` / the exported GLB are unaffected. The durable channel is the paste
block — copy it into `FRAMES[<frame>]["pose"]` / `["joints"]` if it looks right, then
re-render `match` to confirm the gate IoU. Nothing here touches the shared top-level
`SCALE`.

## Notes
- **It can't fix shape.** Like `sweep`, `apply` only re-poses/articulates finished
  geometry. If the object is wrong/detached/under-modelled, edit `build()` first.
- **There is no scale DOF.** `SCALE` is global — one number for the built object —
  so no per-frame verb may move it (N frames would each win a different scale, all
  pasting into the same `SCALE`, last write wins). To resize, edit `SCALE` in
  `scene.py`; to move a frame in DEPTH instead — usually the DOF actually wanted,
  since size and depth trade off in a monocular view — apply `tz`.
