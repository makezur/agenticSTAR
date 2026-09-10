# `analysis.temporal.report` — producing a temporal report from a pose.json

How to turn a run's committed poses into the per-frame temporal report: one row per
frame, every value a single scalar. This is the **consumer-facing** view — the shape
a reader (agent or human) is meant to read. The producer's shape, with vectors and
per-step/per-frame blocks, is `sequence` (its module docstring is its doc); this is
the table over it. The derivation of every number is [maths.md](maths.md).

```bash
cd harness
micromamba run -n artscript env PYTHONPATH=harness \
  python -m analysis.temporal.report \
    --pose-json RUN_DIR/mesh/pose.json \
    [--out /tmp/temporal_report.json] [--order f0,f1,...] \
    [--seams-only | --frames SPEC | --window wNN] [--context N] \
    [--columns vel,accel,JOINT] [--sort-by COL] \
    [--run-dir RUN_DIR | --no-seams]
```

The **table goes to stdout** — that is the report. `--out` additionally writes the
same numbers as JSON (6-decimal precision), for anything programmatic. The view
flags (§5a) scope what is *printed*, never what is *computed*.

---

## 1. Reading the table

```
                                   |         VELOCITY          |       ACCELERATION        | JOINTS (state, velocity, accel)
              step (from->at)  gap |     rot    trans   radial |     rot    trans   radial |    lid_hinge      vel    accel
---------------------------------------------------------------------------------------------------------------------------
                   000100.jpg    - |       -        -        - |       -        -        - |          0.0        -        -
       000100.jpg->000110.jpg   10 |    4.89   0.2950  +0.2872 |  173.98   0.7683  -0.4012 |        -77.0    -77.0    -31.0
       000110.jpg->000120.jpg   10 |  169.19   0.6072  +0.0415 |  171.86   0.6802  -0.1096 |       -185.0   -108.0   +126.7
```

The name column is the **step**, `FROM->AT`. A velocity is a per-step delta filed at
the later frame, so a row names two frames or it names half of a measurement — and
under `--sort-by` the temporal neighbour is no longer the line above, so `from` is
the reader's only route back to the other endpoint. The first frame ends no step and
carries its own name alone. Which of a step's two frames is the wrong one is the
reader's call: **a mis-posed frame inflates the step on each side of it**, so a
single-frame label points one frame past the defect as often as at it.

**What each column is telling you.** Every number is a difference between
**committed poses**, not a measurement of the video. The source video is always
smooth, so structure in these columns is structure in the *poses*:

- **`rot`** — how far the object's orientation turned across the step, as one
  angle. Smooth motion gives a smooth column; an isolated spike is the signature
  of a pose flipped into a different basin (near-symmetric objects), not of the
  object whipping around between two frames.
- **`trans`** — how far the object moved relative to the camera, in fractions of
  its own size: 0.30 means "moved by 30% of its longest dimension in one step".
- **`radial`** — the signed part of that motion along the camera's line of sight
  (`+` = away). Depth is what a single camera constrains **worst**, so pose error
  tends to surface here: an oscillating radial column while the object's apparent
  size in the frames never changes is the poses breathing in depth, not the
  object moving. Radial ≈ 0 under a large `trans` is motion across the view — the
  well-constrained kind.
- **acceleration columns** — where the motion *changed*. Real motion keeps
  acceleration small even when velocity is large; one mis-posed frame between two
  good neighbours is a velocity out-and-back, which lands as a spike exactly on
  that frame.
- **joint columns** — articulation in the joint's own units; `state` is absolute,
  `vel`/`accel` are its first and second differences.

**Anchoring.** A velocity is a per-STEP quantity printed at the step's **later**
frame; an acceleration is per **interior** frame. So the first row has no velocity,
and the first and last have no acceleration. Those cells are `-` — **not measured**,
which is a different claim from `0.0` (did not move). Every joint gets its own
three-column group, so a multi-joint object widens the table (~33 chars per joint).

**What each column is** — with `W_i = R_cam_i R_i` (world orientation) and
`u_i = R_cam_i t_i / s` (camera→object offset, world axes, object units);
[maths.md](maths.md) derives them:

| column | formula | unit |
|---|---|---|
| velocity `rot` | `\|log(W_{i−1}^T W_i)\|` | degrees — a geodesic angle |
| velocity `trans` | `\|u_i − u_{i−1}\|` | object units per step (fractions of the object's longest canonical dimension) |
| velocity `radial` | `û_i · (u_i − u_{i−1})` | object units per step, **signed**: `+` = moving away from the camera |
| accel `rot` | `\|log(Ω_i^T Ω_{i+1})\|`, `Ω_i = W_{i−1}^T W_i` — a **ratio**, folds past 180° | degrees |
| accel `trans` | `\|v_{i+1} − v_i\|` | object units per step² |
| accel `radial` | `û_i · (v_{i+1} − v_i)` | object units per step², signed |
| joint `state`/`vel`/`accel` | absolute state; `Δ_i = state_i − state_{i−1}`; `Δ_{i+1} − Δ_i` | the joint's native units (degrees revolute, canonical prismatic) |
| `gap` | frame-number difference | frames — **reported, never divided by** |

Nothing is divided by `gap`, so a "velocity" is a per-step delta, not per unit time
([maths.md](maths.md) §7 is why).

**Two things the numbers cannot tell you themselves:**

- **The angular acceleration folds.** It is a geodesic angle, so it peaks at 180° and
  a larger change comes back down: a ~340° reversal reports **20°**, the same as a
  mild +10°/+30° ramp. Read it beside the rotation *velocity* — 170 per step versus 10
  and 30 is what separates them.
- **A large rotation is not "wrong" a-priori.** A genuine half-turn and a spurious
  flip into a near-symmetric object's opposite basin produce the *same* number.
  It means "go look at this step", and only visual evidence settles it. When the
  endpoint frames alone make the rotation hard to read, run
  `analysis.viz.pose_pair_sheet` for those two frames. Its posed overlays and
  object-base XYZ triads make the change in facing explicit, which helps decide
  whether the large rotation is physically plausible or a basin flip. The sheet
  is diagnostic evidence, not an automatic verdict.

**Subwindows and gaps.** Reporting on *part* of a run is normal — a refiner reads its
own window — and `--order` scopes it. But frames with a **hole** in them get a loud
warning above the table. The following is a documentation example, not the status
of the window whose brief contains this text:

```
!! NON-CONSECUTIVE FRAMES: 1 gap(s) in the requested order, stride is 10 elsewhere
— 000110.jpg -> 000200.jpg skips 8 frame(s). A velocity across a gap is several
unseen steps summed into ONE number, and an acceleration straddling it differences
that fiction. Those rows are NOT comparable with the rest; check frame_gap on each.
```

It warns rather than failing (the frames may genuinely be all that exist), and
contiguity is judged by the **mode** of the gaps — a run sampled every 10th frame has
gap 10 throughout and is contiguous. `warnings` in the JSON carries the same text.

## 2. What the input has to contain

A `pose.json` as [state_json.md](../../../conventions/state_json.md) §6 defines it.
Only these fields are read:

| field | where | required |
|---|---|---|
| `scale` | top level | yes — the shared object SCALE, a uniform scalar |
| `frames` | top level | yes — `{frame_name: entry}`; names must be numeric-ish (`000120.jpg`) so they can be ordered |
| `object_pose.quaternion` | per frame | **yes** — scalar-first `[w,x,y,z]` |
| `object_pose.translation` | per frame | **yes** — `[tx,ty,tz]`, camera frame |
| `object_pose.scale` | per frame | no — a per-frame echo of the shared scale |
| `camera_c2w` | per frame | see §4 — 4×4 camera-to-world; only its ROTATION is used |
| `joints` | per frame | **yes, if `joint_defs` declares any** — `{joint_name: state}`, absolute; see §3 |
| `joint_defs` | top level | no — but every joint it declares must then be stated on every frame |

`pose` is accepted in place of `object_pose` (the `scene.py` authoring spelling).
`matrix4x4` / `rotation_euler`, if present, are **ignored** and warned about — the
quaternion is authoritative (state_json.md §1).

A minimal three-frame input:

```json
{"scale": 0.5,
 "joint_defs": [{"name": "lid", "type": "revolute", "limit": [-90, 0]}],
 "frames": {
   "000000.jpg": {"object_pose": {"quaternion": [1,0,0,0], "translation": [0,0,2.0]},
                  "camera_c2w": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                  "joints": {"lid": 0.0}},
   "000010.jpg": {"object_pose": {"quaternion": [1,0,0,0], "translation": [0,0,2.1]},
                  "camera_c2w": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                  "joints": {"lid": -15.0}},
   "000020.jpg": {"object_pose": {"quaternion": [1,0,0,0], "translation": [0,0,2.2]},
                  "camera_c2w": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                  "joints": {"lid": -30.0}}}}
```

Two frames is the minimum for any output at all (one step). Three is the minimum for
an acceleration.

## 3. Declared joints must be stated on every frame

This is not a rule this report invents. [JOINTS.md](../../../conventions/JOINTS.md) §3
already states it — *"Every frame must declare a state for every articulated joint.
Omitting one — or omitting `joints` entirely — is a hard error"* — and
`core.joints.require_complete_states` already **is** it, shared with the scene loader,
the pool worker and the sweep engines. The report simply delegates to it:

```
ValueError: temporal report, frame '000010.jpg': joint state(s) 'lid' missing from
the joint states []. Joint states are ABSOLUTE, not increments — there is no value
that means 'leave this joint alone', so an omitted state cannot be defaulted (0.0
would articulate the joint to zero, i.e. straighten it, and render a plausible
picture of the wrong configuration). Name every articulated joint explicitly.
```

**Why the report checks at all, given the loader does.** A `pose.json` is an artifact
on disk: hand-editable, possibly written by an older tool, possibly merged from window
fragments. It reaches this module without passing the loader. Two failures that would
otherwise be silent:

- **Stated on no frame** — the joint vanishes from the report, which is then quiet
  about articulation the model actually has.
- **Stated on some frames** — a gap defaults to REST, so it reads as the joint slamming
  to zero and back: motion nobody authored. The row would even be self-inconsistent,
  `state: null` beside `velocity: 0.0`.

`fixed` joints are exempt (no DOF, identity at any state), and a rigid object with an
empty `joint_defs` reports normally with no joint columns. A state for a joint the defs
do **not** mention is allowed — that is missing metadata (`type`/`limit` come out
`null`), not invented motion.

## 4. Cameras: all of them, or none

`camera_c2w` may be on **every** frame or on **no** frame. Partial coverage is an
error, and it names the offenders:

```
ValueError: 1 of 3 frames have no camera pose (000010.jpg); a sequence needs a
camera pose on EVERY frame (-> pose_frame 'oriented') or on none of them
(-> 'camera'), because residuals taken in the two frames of reference are not
comparable with each other
```

Which case you are in is on the report as **`pose_frame`**, and it changes what the
numbers mean:

- **`oriented`** — every frame has a camera. Rotation is in world frame; translation
  is in world axes measured from each camera's centre. The honest read.
- **`camera`** — no frame has one. Both halves are camera-relative and **conflate the
  object's motion with the camera's**. `radial` is `-` throughout, because without a
  camera there is no camera→object direction to project onto.

Never compare a number from one against a number from the other.

## 5. Window seams

A run refined in **windows** split its frames among several refiner agents, each
working on its own chunk. A **seam** is a step whose two frames were posed by
*different* agents: neither saw the transition, so it is the one place two
individually-plausible windows can disagree with each other.

Seams are read from the newest `RUN_DIR/iterations/NNNNNN/windows/plan.json`. The CLI
defaults `--run-dir` to the pose.json's own run, so seams appear without asking;
`--no-seams` skips the plan entirely.

```
              step (from->at)  gap  win |     rot    trans   radial | ...
       000140.jpg->000150.jpg   10    - |   44.17   0.2185  +0.1920 | ...
>>     000150.jpg->000160.jpg   10  w01 |   18.81   0.0501  -0.0455 | ...
```

`win` names the window that posed the frame (`-` = not in any window, i.e. a
committed frame the plan left alone). `>>` marks a row whose *incoming* step crossed
a boundary. A footer lists them.

**Nothing here consults a magnitude.** A seam is interesting because of *who* posed
it — a 2° seam and a 178° seam are equally unreviewed.

### `--seams-only`: the focused read

Thirty rows is a lot to scan when the question is only "did the windows agree at
their boundaries". This keeps each seam plus `--context N` frames either side
(default 1):

```
temporal report — oriented frame, 30 frames, 8 of 30 rows (seams +/-0)
            ...   5 row(s) not shown
     000150.jpg   10    - |   44.17 ...
>>   000160.jpg   10  w01 |   18.81 ...
            ...   4 row(s) not shown
```

Context matters because a seam number is only interpretable *next to* its
neighbours: 40° at a seam means one thing when the neighbours are 5° and another when
they are 45°. Even at `--context 0` **both** rows of the seam step are kept — the step
spans two frames and one row cannot show it.

Elided runs are always **counted**, including a cut before the first kept row (easy to
miss, usually the largest). A view that hid rows without counting them would read as
the whole table. With no seams at all it says so explicitly rather than printing an
empty table.

## 5a. Focused views — `--frames`, `--window`, `--columns`, `--sort-by`

All of these are **display filters**: the derivatives are always computed over the
whole sequence first, so a row shown in any view carries exactly the numbers the
full table would. Elided rows are counted, same as `--seams-only`.

**Row selections** — pick at most ONE (`--seams-only` / `--frames` / `--window`);
two at once is an ambiguity (union? intersection?) and raises:

- `--frames SPEC` — a comma list of single frames and/or `A..B` ranges over frame
  **numbers**, inclusive, either end open: `000100..000160`, `..000120`,
  `000180..`, `000040.jpg`. A term matching nothing raises rather than silently
  showing an empty stretch.
- `--window wNN` — the rows one refiner window owns (from the same
  global iteration's `windows/plan.json` the seams come from), plus `--context` either side —
  the boundary steps *are* the window's seams, and one row cannot show a step. An
  unknown id raises and names the plan's windows.

**Column selection** — `--columns vel,accel,joints` or a joint's name. Prints only
the named groups; `frame`/`gap`/`win` and the `>>` seam marks always print — they
are the row's identity and provenance, not data a reader opts into.

**Sorting** — `--sort-by COL` with `COL` one of `vel.rot`, `vel.trans`,
`vel.radial`, `accel.rot`, `accel.trans`, `accel.radial`, `<joint>.vel`,
`<joint>.accel`, `<joint>.state`. Rows print biggest-|value| first; unmeasured
rows sink to the bottom (never ranked as 0 — "not measured" is not "did not
move"), and the header says loudly that the order is **not temporal**.

`--sort-by` composes with `--frames`/`--window`/`--columns` ("my window's rows,
biggest rotation first"). **It is refused with `--seams-only`, on purpose:** a
seam is interesting because of *who* posed it, not how big it is — a magnitude
order over seam rows reads the small ones as cleared when nothing cleared them,
and sorting destroys the neighbour adjacency the context rows exist for. The
honest composition is the other way round: a sorted *full* table keeps its `>>`
marks, so a big row that is also a seam says so.

## 6. The JSON form (`--out`)

Same numbers, values only, for programmatic use (schema **6**):

```json
{"schema_version": 6,
 "pose_frame": "oriented",
 "units": {"rotation_deg": "...", "translation_canon": "...", ...},
 "joints_meta": {"lid_hinge": {"type": "revolute", "limit": [-185.0, 0.0]}},
 "frames": [
   {"frame": "000110.jpg", "from": "000100.jpg", "frame_gap": 10,
    "velocity":     {"rotation_deg": 4.893988, "translation_canon": 0.295021,
                     "radial_canon": 0.287175},
    "acceleration": {"rotation_deg": 173.981414, "translation_canon": 0.768311,
                     "radial_canon": -0.401238},
    "joints": {"lid_hinge": {"velocity": -77.0, "acceleration": -31.0,
                             "state": -77.0}}}],
 "totals": {"n_steps": 29, "total_rotation_deg": 1621.7,
            "max_rotation_deg": 175.2,
            "max_rotation_at": {"from": "000300.jpg", "to": "000310.jpg"},
            "max_translation_canon": 1.176,
            "max_translation_at": {"from": "000360.jpg", "to": "000370.jpg"},
            "max_radial_velocity": 1.1575,
            "max_radial_velocity_at": {"from": "000360.jpg", "to": "000370.jpg"},
            ...}}
```

Everything that is not a per-frame **value** is stated once, not per row:

- **`joints_meta`** carries each joint's `type` and declared `limit` — they cannot
  change frame-to-frame, and repeating them N times reads as if they could.
- **No `order` list** — the `frame` column, in row order, IS the order.
- **`from` IS on each row** — a row reports a step, and under `--sort-by` the rows
  beside it are no longer its temporal neighbours, so it cannot be recovered by
  position. `null` on the first frame, which ends no step. (`to` stays off: it is
  the row's own `frame`.)
- **Every `max_*_at` is a `{from, to}` pair**, never one frame — including
  `max_radial_velocity_at`. Radial is promoted into the summary beside rotation and
  translation: it is the DOF one camera constrains worst, hence the likeliest to be
  wrong and the least likely to be caught by eye. It is **signed** (`+` = away),
  ranked by `|value|`, and `null` when no step measured one.
- **Floats are 6 decimals** — far past the table's 2–4, but not the fp noise of a
  17-digit repr.

A missing measurement is `null`, matching the table's `-`.

**Prefer the table unless you need the floats programmatically.** On a real
30-frame run the table is ~1030 tokens against ~4370 for indented JSON — and more
to the point, this data is a matrix, so "which frame is the outlier" is one column
scan in the table (`--sort-by` puts it on the first row) and 30 nested objects in
the JSON.

**No percent-of-travel field.** `delta / (limit[1] - limit[0])` is the fraction a step
consumed, and both operands are on the record — so form it yourself and name it in
your own terms.

## 7. From Python

```python
from analysis.temporal import report

r = report.build("RUN_DIR/mesh/pose.json",
                 run_dir="RUN_DIR")             # run_dir -> seams are marked
print("\n".join(report.table_lines(r)))          # the same table
print("\n".join(report.table_lines(r, seams_only=True, context=2)))
print("\n".join(report.table_lines(r, window="w03")))            # one refiner's rows
print("\n".join(report.table_lines(r, frames="000100..000160",
                                   columns="vel", sort_by="vel.rot")))

for row in report.seam_rows(r):                 # just the boundary rows
    print(row["frame"], row["window"], row["is_seam"])

for row in r["frames"]:
    v = row["velocity"]["rotation_deg"]
    if v is not None and v > 90:
        print(f"{row['frame']}: {v:.1f} deg — go look")
```

`--order f0,f1,...` (or `order=[...]`) overrides the frame sequence; the default is
temporal order by numeric frame index, so a `pose.json` authored out of order still
reports along the timeline.

## 8. The intended loop

1. Run this report on the poses a window owns (`--seams-only` when the question is
   the boundaries).
2. Before looking, write in `NOTES.md` which steps you *expect* to be large and why
   (a hinge closing, a half-turn in the video). That written expectation is what you
   compare the numbers against; without it a plausible number always looks fine.
3. Read the table. A big step you did not expect is a suspect flip; an expected
   large move that measures small deserves a second look. A large acceleration means
   the motion changed hardest there — an out-and-back is the shape of one frame
   sitting in the wrong basin between two neighbours that agree with each other.
   **For rotation, read the acceleration beside the two legs' own `rot` velocity,
   not alone** — it peaks at a half-turn and folds past it ([maths.md](maths.md) §3).
4. **Look** — the two source frames *and* their committed renders
   ([seam_sheet](../viz/seam_sheet.md)). The source video is always smooth, so only
   a render shows which basin a pose landed in.
5. When the pictures and the residual **disagree** — the numbers say a half-turn
   and the frames show none — that is a pose to **fix**, not a number to explain.
   Re-pose to the basin the images support and run the report again.
6. Repeat per window. Nothing in the tooling clears a step for you: the numbers
   locate the question and the frames answer it.

## 9. Errors you should expect

| message | cause |
|---|---|
| `N of M frames have no camera pose (...)` | partial `camera_c2w` coverage — §4 |
| `frame 'X' has no pose quaternion` | a frame's pose is missing. **Not** defaulted to identity: identity is a plausible-looking pose, so the residual would come out large and legible instead of failing (state_json.md §8) |
| `frame 'X' nests 'joints' INSIDE its pose` | `joints` is a sibling of the pose, never nested; nested it would read as *no* joints, i.e. every joint silently at rest |
| `frame 'X': joint state(s) '...' missing` | §3 — a declared joint has no state on some frame; a gap would report articulation nobody authored |
| `scale must be one numeric scalar` | anisotropic or sequence `scale` |
| `UserWarning: ... derived cache key(s) matrix4x4` | harmless, the keys are ignored |

All of these fail loudly rather than producing a plausible-but-wrong number.

## 10. Where the numbers come from

```
pose.json
  -> analysis.lib.pose_read          ingest + the state_json.md gates
  -> analysis.pose_diff              lift each frame ONCE (camera rotation applied
                                     here), then the residual between two placements
  -> analysis.temporal.sequence      pairs the track into steps; adds derivatives
  -> analysis.temporal.report        THIS: selects the scalars into a per-frame table
```

This module **computes nothing** — every value is already a scalar on the sequence
report, taken where the vectors were. So a consumer and the producer cannot disagree
about a number.
