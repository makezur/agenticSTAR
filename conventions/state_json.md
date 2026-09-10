# state_json.md — how tools serialize a frame's STATE to JSON

A frame's state is **pose + articulation**; both are here, because both are written
by the same writers into the same records at the same precision.

[POSE.md](POSE.md) says what a pose *is*, [JOINTS.md](JOINTS.md) what a joint state
*is*. This says how they are written and read. If this disagrees with the code, the
code wins.

---

## 1. The payload

```json
{"quaternion": [w, x, y, z], "translation": [tx, ty, tz], "scale": 1.0}
```

- **`quaternion`** — scalar-first, unit. **Authoritative**: R comes from this and
  nothing else.
- **`translation`** — camera-0 frame, scene units, applied after scale.
- **`scale`** — uniform scalar, **optional**, only ever an echo of the shared
  `SCALE` (§4).

Nothing else. `matrix4x4`, `rotation_euler`, `euler_order` must **not** be written —
they were derived caches that went stale and won over the quaternion. Derive the 4x4
via `lie.object_pose_matrix`; strip them when reading old files
(`multiagent.windows._DERIVED_CACHE_KEYS`). `rotation_euler` is still *accepted* on the
authoring path (`transforms.rotation_matrix_of`, pre-quaternion scenes), with
`quaternion` winning when both are present.

Why keep it: q/t/s are exactly the DOFs of `T·R·S`, so no two fields can disagree,
and scalar-first matches both `mathutils` and three.js — nothing reorders.

---

## 2. Articulation — `joints` is a SIBLING of the pose

```json
{"pose": {"quaternion": [w,x,y,z], "translation": [tx,ty,tz]},
 "joints": {"door_hinge": -80.0, "drawer_slide": 0.25}}
```

`joints` maps `joint_name -> state` (revolute degrees, prismatic canonical units).

1. **Sibling, never nested** inside the pose object — a joint state is not part of
   the rigid placement.
2. **Complete or absent.** States are absolute, so an omitted joint does not mean
   "unchanged" — defaulting it to `0.0` *straightens* it. Write `joints` at all and
   you write every declared joint; say nothing about articulation by omitting the key.
3. **6dp**, like the pose (§5), via `core.state_json.round_joints`.

Omitting is correct only where articulation is provably untouched:
`osweep`/`oapply`/`oapply_all` never search it, so their fragment carries pose alone
and a merge leaves committed joints as they were.

---

## 3. Key name — authority

| key | means | written by |
|---|---|---|
| `object_pose` | **committed** — the run's current estimate | `rig/scene.py` (`mesh/pose.json`), `multiagent.windows` (`merged_poses.json`) |
| `pose` | **proposed** — candidate, fragment, paste block | sweep reports, `<stem>_poses.json`, pool orders, reseed entries |

A proposal becomes committed only through `multiagent.windows merge`/`apply` or a paste
into `scene.py` — never by editing `pose.json` in place. A reader that accepts both
spellings must say why (`analysis/pose_diff.frame_entry` does: a diff is valid on
either).

---

## 4. Scale

`scale` belongs to the object, not the frame:

- **top level**, once, in any whole-run document (`pose.json`, `merged_poses.json`)
  — authoritative;
- **per-frame echo** only where a consumer needs a self-contained placement
  (`pose.json` frame entries, report `best`/`ranked[]`/`base_pose`);
- **absent** from a per-frame proposal, so a fragment or paste block cannot smuggle
  a scale change into a merge.

Read precedence (`lie.pose_from_object_pose`): the pose's own `scale`, else the
document's, else `1.0`. Anisotropic scale is rejected at every boundary.

---

## 5. Precision

**6 decimal places** for quaternion, translation, and joint states
(`core.state_json.round_pose` / `round_joints`) — lossless at the pixel level,
diffable, and a round-trip through `scene.py` is a no-op.

A `.txt` paste block prints the SAME values its `.json` carries
(`state_json.paste_literal`). 6 *significant* digits is not the same rule: it
narrowed `12.345679` to `12.3457`, so the pasted winner was not the scored pose.

6dp leaves a unit quaternion non-unit at ~1e-6. Handled once at the READ boundary
(`transforms._unit_quat` / `lie.check_quat_norm` renormalize, and warn when the norm
is far from 1). **Writers must not compensate.**

`frame_pose_to_dict` writes full precision — it serializes a decomposed matrix and
is the committed record. Every proposal path rounds.

---

## 6. Containers

Same §1 payload in all of them; only the envelope differs.

**Committed — `mesh/pose.json`, `merged_poses.json`** (`rig/scene.py`)
```json
{"frame_convention": "camera0_opencv (+X right, -Y up, +Z into scene)",
 "canonical": "+Z up, centered at origin, longest dimension ~= 1",
 "scale": 1.0, "reference_frame": "000000.jpg", "parts": [], "joint_defs": [],
 "frames": {"000000.jpg": {"moved": false, "object_pose": {}, "joints": {},
                           "view_index": 0, "camera_c2w": [], "intrinsics": {}}}}
```

**Fragment — `<wid>/poses.json`, `<stem>_poses.json`** (`multiagent/windows.py`)
A subset proposal, gated on `scene_sha1` (a mismatch is fatal). `merge` reads only
`pose.quaternion`, `pose.translation`, `joints` (overlaid, not replaced).
```json
{"schema_version": 2, "scene_sha1": "<sha1 of scene.py>", "window_id": "w0",
 "expected_motion": [{"frames": "000010..000040", "expect": "lid opens"}],
 "frames": {"000010.jpg": {"pose": {}, "joints": {},
                           "confidence": "high|medium|low", "notes": "why"}},
 "step_calls": [{"from": "000010.jpg", "to": "000020.jpg",
                 "verdict": "coherent|unsure", "evidence": "what you SAW",
                 "pose_hash": "<stamped by selfcheck>"}],
 "report": [{"frame": "000012.jpg", "note": "saw but cannot fix"}],
 "model_feedback": []}
```
Per-frame free text is `notes`; the window-level list is `report[].note`.

`expected_motion` and `step_calls` are required of a **refiner's** fragment — one
carrying a `window_id`, i.e. one an agent looked at. The sweep family's
`<stem>_poses.json` has no `window_id` and owes neither: nobody inspected those
steps, so `merge` routes them up to the orchestrator instead. Verdict vocabulary
and the freshness rule live in
[multiagent/calls.py](../harness/multiagent/calls.py) and
[multiagent/freshness.py](../harness/multiagent/freshness.py); §9 says why the
hash covers the shared scale.

**Report — sweep family `<stem>.json`** (`metrics.py`, `shared_engine.py`)
`base_pose`, `best`, `ranked[]`, each with `pose` (carrying `scale`) and/or
`joints`. Read by `pool/reseed.py`, never merged directly. The multi-frame verbs put
the same entries in a `frames` **list** plus the canonical `rotation` that produced
them, and emit a sibling fragment so the result *can* be merged.

They also carry `pivot` (the orbit point a recorded pose `order` turned about — shape
in [harness/views/sweeps/sweep.md](../harness/views/sweeps/sweep.md)). It is **not**
§1 payload: nothing pastes it, it is the frame of reference an `order` is expressed in.
It does follow §5 — 6dp via `round_component`, its `joint_states` via `round_joints` —
so a pivot never prints to a different precision than the pose it explains.

**Order — pool** (`pool/reseed.py`)
One proposed `pose` + `joints` + `reseed` provenance, so a chain of hops is a chain
in the data.

---

## 7. Rules for a new tool

1. Write the §1 payload — no invented fields, no matrix.
2. Key by authority (§3): not the committed file's single writer → write `pose`.
3. Round via `core.state_json.round_pose` / `round_joints` (§5).
4. Read rotation via `lie.pose_from_object_pose`; read scale by its precedence.
5. Crossing a scene boundary carries `scene_sha1`.

---

## 8. A missing rotation is an error, never identity

No write path defaults a pose. `round_pose` indexes `quaternion`/`translation`
rather than `.get`-ing them, `metrics._quat_of` raises, and
`pool/reseed._pose_block` refuses `rotation_euler` instead of guessing a
conversion — a defaulted pose publishes a rotation nobody scored, indistinguishable
downstream from a real result.

One read path defaults on purpose: `lie.pose_from_object_pose` treats a missing
`quaternion` as identity, which is right for a partial hand-authored `scene.py` pose.
That default is **scoped to a single pose dict**, not a document: reading a frame out
of a whole-run document goes through `analysis.lib.pose_read.frame_entry`, which
RAISES on a missing `quaternion`/`translation` (opt out with `require_pose=False`
where a partial record is legitimate — a hand-authored frame, an in-progress
fragment). Identity is the worst substitute precisely because it is *plausible*: a
residual against it comes out large and legible, so a frame that lost its pose reads
as a frame that rotated. `pose_read` is also where §2's sibling-`joints` rule is
enforced — nested, `joints` reads as no joints, i.e. every joint silently at rest.

---

## 9. A temporal verdict is hashed against the poses it judged — scale included

A temporal verdict ("this 47 deg step is the lid being thrown open, and you can see
the hinge clear the rim") is a claim about two SPECIFIC poses. Move either and the
sentence still reads fine while no longer being about anything in the trajectory, so
each verdict carries a `pose_hash` (`multiagent/freshness.py`) recomputed at
validation time: mismatch = stale = uncalled.

The hash covers both frames' quaternion + translation, both frames' **joint states**,
and the run's **shared `scale`**:

- **joints**, because an articulation change moves the object on screen exactly as a
  base-pose change does — a re-hinged frame whose base pose never moved must stale
  its verdict too;
- **scale**, because a rescale changes NEITHER frame's pose numbers (§4: `scale`
  lives on the object, not the frame) and still changes every rendered silhouette and
  every canonical residual. Either it is in the hash or a rescale silently keeps every
  temporal verdict in the run.

Deliberately NOT hashed: the measured residual. The question is "are these still the
poses I judged", not "did the numbers change" — hashing a measurement would let a fix
to the residual maths stale every verdict in the run without a pose having moved.
Quantization is §5's 6dp (so a hash survives a JSON round trip) and quaternion sign
is canonicalized (q and -q are one rotation; staling for a no-op would teach everyone
to re-stamp without re-looking).

---

## Source of truth

- Shared helpers (precision, `round_pose`, fragment version):
  [core/state_json.py](../harness/core/state_json.py).
- Committed writer + authored-rotation reader:
  [rig/transforms.py](../harness/rig/transforms.py).
- Pose core + read precedence: [rig/lie.py](../harness/rig/lie.py).
- `mesh/pose.json`: [rig/scene.py](../harness/rig/scene.py).
- Fragments + merge: [multiagent/windows.py](../harness/multiagent/windows.py).
- Temporal verdicts + freshness (§9): [multiagent/calls.py](../harness/multiagent/calls.py),
  [multiagent/freshness.py](../harness/multiagent/freshness.py).
- Sweep reports: [views/sweeps/lib/metrics.py](../harness/views/sweeps/lib/metrics.py).
- Reading every dialect back: [pool/reseed.py](../harness/pool/reseed.py).
