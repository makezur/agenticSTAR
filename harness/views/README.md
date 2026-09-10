# `views/` — the render tools

Each view is selected via `harness/render.sh RUN_DIR/scene.py RUN_DIR/views
--views <name>[,<name>...]`. One `render.sh` launch runs all requested views in
order (they share the single Blender process). Each view has a `.md` next to its
`.py` — that doc is the single source of truth for how to use it.

The five candidate-report verbs (`sweep`/`apply`, `osweep`/`oapply`/`oapply_all`)
live in [`sweeps/`](sweeps/), each a `.py` + `.md` pair like every other view; the
engines and planners they share are one level down in `sweeps/lib/`. The rest are
plain renderers directly in `views/`. All nine are peers to a caller — the table
below is the list that matters, and the directory split is an implementation one.

Each `render.sh` call writes into a fresh `RUN_DIR/views/<NNNN>/` pass dir and
PRINTS that dir; read renders (and write derived artifacts) through the pass dir
`render.sh` prints. A view not rendered this pass is absent from it (no stale
leftovers).

| view | answers | when to run | doc |
|---|---|---|---|
| `match` | does my build + per-frame pose + joint state overlay each photo? (`match_<frame>.png`) | every iteration | [match.md](match.md) |
| `turntable` | is the object a coherent articulated assembly in EVERY state, from every angle including above and below? (read `turntable_sheets/turntable_sheet_<state>.png`, one page per state; tiles are `turntable_<state>_az<AAA>_el<±EE>.png`) | every iteration | [turntable.md](turntable.md) |
| `mechanism` | *(module — present only when the run's `modules.json` enables `mechanism`; see core/modules.py)* does each JOINT move the way I declared it? (read `mechanism_sheets/mechanism_sheet_<joint>.png`, one page per joint with two picked-viewpoint strips; tiles are `mechanism_<joint>_r<R>_s<NN>_<state>.png`) | every iteration, and **always after authoring or changing a joint's `axis`/`origin`/`limit`/`child`** — the turntable holds articulation and so is nearly blind to a wrong `axis` SIGN | [mechanism.md](mechanism.md) |
| `ids` | which primitive owns this region? | when a region is wrong and you're unsure which part | [ids.md](ids.md) |
| `sweep` | which **pose and/or joint states** best fit the mask? (SEARCH) | to auto-dial pose (R,t — SCALE is global, never swept) and/or a moved frame's joint DOF(s) — name pose orders, joints, or both in one `--sweep-ranges`; a wide `yaw`/`pitch` range + `--sweep-dump-topk` handles unknown facing | [sweeps/sweep.md](sweeps/sweep.md) |
| `apply` | what does the frame look like — and score — with THIS order applied? (imperative: a 1-candidate sweep) | to CHECK a specific tilt/turn/open — `--apply 'pitch:90;door_hinge:70'` composes the order onto the frame's pose, renders + scores it vs the mask, with a paste-ready block | [sweeps/apply.md](sweeps/apply.md) |
| `osweep` | which canonical rotation does EACH frame want? (SEARCH, object-centric, PER FRAME) | when a frame reads wrong in a way named in the object's own axes ("facing backwards") — right-multiplies a canonical rotation vector (`rx/ry/rz`, object's own axes) onto that frame's pose; every frame searches the same grid independently and keeps its own argmax; needs `--masks-dir` | [sweeps/osweep.md](sweeps/osweep.md) |
| `oapply` | what does each frame look like — and score — with THESE canonical rotations? (imperative, PER FRAME) | `apply`'s object-centric twin: `--oapply 'flips'` / `--opreset 'z:quarters'` renders every candidate for every frame and each frame picks its own winner — the verb when the frames DISAGREE (a window seam); needs `--masks-dir` | [sweeps/oapply.md](sweeps/oapply.md) |
| `oapply_all` | what do ALL frames look like — and score — with THIS ONE shared canonical rotation? (imperative, WHOLE RUN) | when the built object is mis-oriented in EVERY frame (flipped/on-its-side) and one decision commits it — `--oapply 'rz:180'` right-multiplies it onto every frame, renders + scores each, paste-ready per authored frame; needs `--masks-dir` | [sweeps/oapply_all.md](sweeps/oapply_all.md) |
| `crop` | is the object cut off by the frame? | when the object may be too big / off-center | [crop.md](crop.md) |
| `visibility` | which bits are visible, where, what's in front? | on demand when a part is missing/hidden (N+1 renders) | [visibility.md](visibility.md) |
| `depth` | does my geometry sit at the right depth per frame? (`depth_<frame>.npy`) | every iteration a Pi3X pointmap is available (the depth-blindness cure; cheap, same launch as match/turntable) | [depth.md](depth.md) |

`match` + `turntable` + `mechanism` are the cheap every-iteration set — `match` runs
once per frame, `turntable` once per unique articulation state, `mechanism` once per
declared joint — and `depth` **joins them every iteration when a Pi3X pointmap is
present** (all ride one EEVEE launch). `turntable` and `mechanism` are complementary
halves of the shape gate: the turntable moves the CAMERA and holds articulation;
`mechanism` holds the camera and sweeps the JOINT — only it can catch a mirrored
`axis` (see [mechanism.md](mechanism.md)).
`ids`, `sweep`, `apply`, `crop`, and `visibility` are single-frame
diagnostics — they run against the frame chosen by `--frame` (default: the
reference; point it at the moved frame for a joint or coupled `sweep`, or the frame
you want to re-pose with `apply`). The object-centric trio are **multi-frame** views
(like `turntable`): they score across ALL `--frames` in one call and ignore `--frame`.
Within them the SCOPE is the thing to get right — `osweep`/`oapply` fit each frame
**independently** (N independent `sweep`/`apply` runs sharing one Blender process),
while `oapply_all` commits **ONE** rotation to every frame. There is deliberately no
whole-run *searched* mode: a shared rotation ranked by MEAN IoU can be mediocre in
every frame, so the whole-run path is imperative only (a panel you look at).
All of them restore the scene afterward, so they never affect the
`match`/`turntable` outputs or the exported GLB. What the three SHARE — the
canonical DOFs, the `--opreset` spelling, the panel/visual budget, and the paste
contracts — is in [sweeps/object_rotations.md](sweeps/object_rotations.md); each verb's doc
covers only what is different about it.

Flag semantics: run `harness/render.sh` with no scene (or read
`../rig/args.py`) — each flag's `--help` string is the ground truth.

## Adding a view
Drop a module exposing `render_view(ctx)` (ctx is a `rig.render.Context`),
register it in `__init__.py`'s `VIEWS`, and add a `<name>.md` here.
