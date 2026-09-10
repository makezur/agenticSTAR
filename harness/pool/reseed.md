# `reseed` — save a pose+joints pick, then spend it as the next hop's seed

Fitting one frame is **coordinate descent**, not one big search. A 7-DOF grid dense
enough to matter is unaffordable, so the loop is: sweep two or three DOFs densely,
**pick a candidate by eye**, sweep the next few *starting from that pick*, repeat.
Each hop is cheap; the hops compose into a fit no single affordable grid reaches.

The harness had every step of that loop except **saving the pick**. A sweep never
wrote its winner anywhere — `sweep.json` is a report, and nothing consumed it — so
"start from the pick" meant retyping a quaternion, a translation, and every joint
state into the next order.

Two things go wrong with that.

**The revert.** Placement and articulation travel in two *different* order keys:
`sweep.pose_start` (or a top-level `pose`) and a top-level `joints`. And `pose_start` is
placement-**only** — `transforms.normalize_placement` builds a fresh dict of
rotation/translation/scale, so a `joints` key handed to it is silently dropped.
Carry the placement, forget the joints, and the joint reverts to the frame's *committed*
state while the pose carries. Every candidate in the next hop is then wrong, and
every render still looks entirely plausible.

**The transcription.** ~10 float literals per hop, copied from a text report into
JSON. The cost isn't the typing; it's that this is the step where a hop gets quietly
dropped.

So a reseed **always carries pose and joints together**. There is no flag to carry
one without the other — that is the whole op.

## Two constraints shaped it

**The pick is rarely the winner.** The workflow is built on *overruling* the numeric
winner: IoU is a documented fast proxy, and a silhouette-best candidate is routinely
an edge-on trap. "That order's best" is therefore useless for exactly the hops that
matter, so addressing an **arbitrary** candidate is a requirement, not a nicety.

**Five JSON dialects already spell the same pose+joints pair.** `mesh/pose.json`
(`object_pose`/`joints`), a sweep report's `best`/`ranked[i]` (`pose`/`joints`), a
sweep report's `pose_start` (a bare placement), a `multiagent.windows` fragment /
`progress.json` / `poses.json` entry (`pose`/`joints` + confidence/notes), and a pool
order (`pose` **or** `sweep.pose_start`, plus a top-level `joints`). A tool that hardcoded
one would work for one hop and break at the seam between phases. So this is **one
normalized record with a reader per dialect and a writer per dialect**; a sixth
format later is one reader, not a rewrite.

## The `seed` order key (the common case)

Drop this in the spool and the manager expands it at claim time — you never see the
floats:

```json
{"id": "w04-000250-b2", "views": "sweep", "frame": "000250.jpg",
 "seed": "RUN/spool/renders/w04-000250-b1/sweep.json#rank=14",
 "sweep": {"ranges": "yaw:-6,6,7", "refine": 1, "dump_topk": 6}}
```

`seed` fills the placement **and** the full `joints` state from candidate rank 14 of
the previous order. Every sweep's `sweep.txt` ends with a ready-made **NEXT HOP**
block in exactly this shape, with the rank left as a placeholder for you to fill.

**Which key the placement lands in depends on `views`**, because the two channels do
different things and only one of them exists for a given verb:

| `views` | key | what it moves |
|---|---|---|
| `sweep` / `apply` only | `sweep.pose_start` | **where the grid starts** — read by `views/sweeps/lib/engine` and nowhere else |
| anything else (`osweep`, `oapply`, `oapply_all`, `match`, a mix) | top-level `pose` | the **frame**, via `serve.py` onto the resolved FrameSpec — so it reaches every view |

`pose` is the general channel: a sweep's start is derived from the frame
(`ctx.placement = frame_placement(fs.pose, scale)`), so moving the frame moves the grid
too. `sweep.pose_start` is the narrower one — it moves *only* where the search departs
from, which is what you want when the frame's committed pose should stay as it is.
Writing `pose_start` for a verb that never reads it would emit a placement the run
discards while the joints applied alone: the revert again, one key at a time. `_placement_key_for` makes the choice once,
for both the `seed` key and `--as order`.

The canonical verbs are seedable for the same reason they render at all: `osweep`
composes `s·(R·dR)·p + t` onto the frame's pose and renders its joint states — it just
doesn't *search* over them.

- An explicit `sweep.pose_start`, `pose`, or `joints` **beside** a `seed` wins — that's how
  you override one half (nudge the placement, keep the seeded articulation) without
  restating the other. Same contract as the manager's other claim-time fills
  (`resolve_hand_mask`, `resolve_object_masks`): a key the order already carries has
  decided.
- Unlike those fills, an unresolvable `seed` **fails the order** rather than
  degrading to "not filled". Their fallback is merely worse scoring; a seed's
  fallback would be the frame's committed pose — a plausible render of the wrong
  configuration.
- `seed` is replaced by `seed_from` on the claimed order, and a `seed` line goes in
  `ledger.jsonl`. The claimed order is deleted on recycle, so the ledger is what
  makes a multi-hop descent auditable afterwards.

## Addressing: `<path>#<selector>`

One syntax for "which record in that file", across every dialect.

| selector | resolves to |
|---|---|
| `#rank=14` | the `ranked` entry whose **`rank`** is 14 |
| `#candidate_id=c1` | the entry with that `candidate_id` |
| `#best` | the numeric winner (same as omitting the selector) |
| `#000250.jpg` | that frame, in a frame-keyed file or a multi-frame report |
| *(omitted)* | the sole record, if the file holds exactly one; otherwise refuses and lists them |

**`rank` is a candidate's name, not its line number.** The listing is gappy: `topk`
cuts it, then any candidate that got a panel is appended keeping its true rank — so
the 11th entry is routinely `rank: 13`. `sweep.txt` prints both a `row` column (line
number) and a `rank` column, and `rank` is the one to quote and to feed here.

**Known limit:** `topk` (default 10) bounds `ranked`, so a candidate outside the top-k
is not addressable at all. Raise `topk` on a sweep whose candidates you may want to
descend from. A rank miss says so and lists what *is* available.

## Every record is a whole joint state

Each candidate's `joints` is the **full absolute state** — every declared joint,
whether the grid varied it or held it — and each `frames[]` entry of an
`osweep`/`oapply`/`oapply_all` report names the articulation that frame was rendered
at. So reseeding *reads* a state; it never assembles one.

`meta.held_joints` stays in a sweep report for the human "Swept: … Held: …" line.
Nothing reconstructs state from it.

That's a consequence of the asymmetry this op exists for:

- a camera-frame **pose** DOF (`roll`/`yaw`/`pitch`/`dpx`/`dpy`/`tz`) is an
  **increment** onto a base placement. `0` *is* the hold; omitting it is a lossless
  no-op.
- a **joint** state is **absolute**. `0` means "articulate to zero" — a real move.
  There is no value meaning "leave this alone", so an omitted joint state cannot be
  defaulted, only **refused** (see `core.joints.require_complete_states`).

## CLI

```bash
# save a pick, addressed by the rank you saw in sweep.txt
python -m pool.reseed --from RUN/spool/renders/w04-000250-b1/sweep.json#rank=14 \
                      --to RUN/iterations/000002/windows/w04/progress.json#000250.jpg

# spend it: emit an order body that descends from a saved state
python -m pool.reseed \
    --from RUN/iterations/000002/windows/w04/progress.json#000250.jpg \
                      --as order --frame 000250.jpg \
                      --sweep-ranges 'dpy:-0.02,0.02,5' --id w04-000250-b2
```

`--as` also takes `entry` (a fragment/progress frame entry), `placement` (the bare
pose), and `record` (the normalized form, for debugging). `--to` read-modify-writes
the target so other frames' entries and the refiner's own `notes`/`confidence`
survive, and records the source in the entry's `reseed` block — which the reader
picks back up as `provenance.parent`, making a chain of hops a chain in the data.

`--frame` is an **assertion**, not a resolver: it refuses a record belonging to a
different frame, which is a plausible copy-paste slip when one window drives eight
frames.

## Crossing frames on purpose — `seed_cross_frame`

Frame-scoping is the default because seeding frame N from frame N−1's record by
copy-paste is a real slip.

But "start this frame from the neighbour I just fitted" is a legitimate and common
move — on a hand-held capture a 10-frame gap is a small screen-space step, and the
neighbour's *fitted* pose is a far better grid centre than this frame's *committed*
one (which may be the thing you are fixing). So the refusal is **waivable, per
order**:

```json
{"id": "w05-000350-a1", "views": "sweep", "frame": "000350.jpg",
 "seed": "RUN/iterations/000002/windows/w05/progress.json#000340.jpg",
 "seed_cross_frame": true,
 "sweep": {"ranges": "dpx:-0.08,0.08,7;dpy:-0.08,0.08,7;tz:-0.15,0.15,7"}}
```

**Verbatim** is the contract: the pose lands exactly as the source held it, and the
inter-frame **camera motion is not compensated**. The sweep has to absorb that step,
so give it the range to — a cross-frame hop usually wants placement DOFs (recipe D)
in the same order. Reseating through `cameras.npz` would be a different op
(`transforms.reseat_pose` does that math for the render path); doing it silently here
would make `sweep.pose_start` a pose the source never held, which is exactly the kind of
"plausible but not what you picked" this module exists to prevent.

The waiver is recorded, because a cross-frame hop is the most likely cause of a bad
seam: the flag survives on the claimed order, the manager logs the crossing, and the
`seed` ledger event carries `cross_frame: "<source frame>"`. It waives **only** the
frame check — a selector miss, a bad path or a malformed pose still fails the order.

## Scale is never carried

A record's `pose` carries `scale` only when its source dialect had one — a **report**
entry does (it is a self-contained placement), a **fragment / progress entry / paste
block** does not, by design (state_json.md §4: a proposal must not be able to smuggle
a scale change into a merge). Reseed passes that absence **through** and never
invents a number.

The engine supplies it instead: `views/sweeps/lib/engine._resolve_pose_start` stamps the
frame's shared `SCALE` onto whatever `pose_start` arrives, and refuses one that states a
*conflicting* one. A defaulted scale would render every seeded hop at the wrong size
with plausible renders and healthy-looking IoU, so `normalize_placement` raises on a
missing scale.

Pure Python — no `bpy`, no `cv2` — so it runs in the analysis env and the manager can
call it at claim time.

## Not doing: teaching `pose_start` to carry joints

`sweep.pose_start` is a *placement*, and `normalize_placement` dropping a stray `joints`
key is correct. Widening it to sometimes-carry articulation would make one key mean
two things depending on who wrote it. The fix is a key that means "both, from one
place" — which is `seed`.

## Limit: coupled valleys

Descent moves one DOF group at a time, so it cannot escape a **coupled** valley —
base orientation trading off against a hinge, where any single-group move scores
worse and only a joint move scores better. Sweep those together (`"space": "both"`,
recipe C) instead of alternating hops.

See also: [sweep.md](../views/sweeps/sweep.md) (the report and its `rank` column),
[README.md](README.md) (the spool protocol), and
[briefs/core.md](../multiagent/briefs/core.md) § Per-frame loop (the refiner's
loop, step 4).
