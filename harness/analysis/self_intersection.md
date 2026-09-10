# `self_intersection.py` — part-pair overlap volumes, a warning beside the temporal report

Two watertight parts of a real object cannot share volume — a part-pair overlap
is direct evidence the pose or articulation is wrong. But small overlaps are
also how touching parts are *modelled* (a label sunk into the body), so this
tool **reports and marks, never judges**: no exit-code gate, no threshold. `!!`
is a display mark with its floor printed in the header.

**`shape_pass.sh` runs it on every pass, unconditionally.** Inputs are what a
pass already produces (its `pose.json` + exported GLB), ~1 s, no renders.
Output: `PASS_DIR/self_intersection.txt` (+ `.json`). A non-zero exit means the
report could not be *produced* (unreadable GLB, incomplete joint states), which
fails the pass like any other hard scorer.

Reading it is the orchestrator's job, never a pose-refiner's (part-pair
plausibility is a property of the merged trajectory). Read the onsets beside
`adjudicate`'s temporal reads.

## Run it directly (artscript env, from `harness/`)

Rarely needed — every pass writes one. For re-reading a pose.json without
rendering (`--pairs`, a hand-edited state):

```bash
micromamba run -n artscript env PYTHONPATH=harness python -m analysis.self_intersection \
    --pose-json RUN_DIR/mesh/pose.json --pairs
```

The GLB defaults to `object.glb` beside the pose.json. The default view is one
row per frame in temporal order, so it reads beside the temporal report:

```
          frame  new pairs    ovl  worst%  worst pair (side)
------------------------------------------------------------
!!   000100.jpg   11    11  0.067  100.00  dispensing_nozzle(0.07) ∩ lid_disc(0.17)
     000160.jpg    -     -      -       -  -
```

## Reading the numbers

| column | meaning |
|---|---|
| `worst` / `frac_smaller` | overlap volume / the smaller part's volume — the "is this plausible" number: 0.2 % is a graze, 5 % is a lid inside a body |
| `ovl` / `size` | the overlap volume's cube root (equivalent-cube side, canonical units) — read against the part sides on the pair label |
| `(n.nn)` on part names | that part's own equivalent-cube side — the yardstick for `ovl` |
| `new` | pairs whose overlap **appears** at this frame; `-` means no overlap, never 0 |
| `overlapping at` | frame spans; each span's first frame is the **onset** — the step *into* it drove the parts together, the row to read in the temporal report (JSON: `onsets`) |

Design contacts (pairs inside one rigid group, whose overlap articulation can
never change) are computed once and stated in the header, never marked.

## Mechanics

Computed in the canonical frame from joint states only (`rig.fk`, numpy — the
same FK the renderer uses), so identical joint states are computed once. Exact
mesh booleans via `manifold3d`, AABB-pruned; ~1 s for a 30-frame 16-part run.
The price is watertightness: a **non-watertight part is excluded from every
pair** and listed in a geometry table with its diagnosis (open edges /
non-manifold / winding; JSON: `skipped_parts`). Its overlaps are unknown, not
zero.

`unknown_pairs` (JSON) counts articulating pairs with an excluded part on one
side; while it is non-zero the footer refuses `NO articulating-pair overlaps`
and names the excluded parts that ride a joint — an excluded joint child means
the one pair that could expose a bad hinge was never checked. Fix the parts and
rerun before reading a clean report as clean.

Hand–object or object–scene collision is out of scope: the object's own parts
only.
