# `turntable` view — 3D shape gate (required every iteration, EVERY state)

**Answers:** "does the object have coherent 3D shape and part placement from every
angle — including above and below — in every observed articulation state?"

A set of orbiting **camera** views around the object (a genuine camera orbit),
rendered **once per unique articulation state** across the frames (so a hinge that
is closed in one frame and open in another is checked in both). It vets **shape**,
not pose, and it is **not an optional sanity-check**. The match view and silhouette
IoU are **depth-blind**: a part at the wrong depth can overlap the silhouette
perfectly. The turntable exposes that, along with malformed hidden-side geometry
and gross intersections. Treat visibly incoherent geometry in any observed state as
a hard failure regardless of IoU.

## What you read: one sheet per state

The shape pass ([shape_pass.md](../utils/shape_pass.md)) builds a **sheet per
articulation state** — that is what to open:

```
PASS_DIR/turntable_sheets/turntable_sheet_<state>.png
PASS_DIR/turntable_sheets/turntable_sheet_manifest.json
```

Every orbit view of that state is on one page, opened by the state's **SOURCE
photo** as a reference tile (identity anchor — no render on the page is
pixel-aligned to it), each render tile labelled with the direction it was shot
from and grouped **FROM ABOVE / LEVEL / FROM BELOW**. The header names the state
and the source frame it was labelled by, so a coherence failure leads straight
back to the photo. Read **every state's page** each iteration and once more
before finalizing — a high IoU never clears this gate on its own.

The individual tiles stay on disk beside the sheet; open one when a suspect region
needs full resolution. To build sheets by hand for any pass dir, see
[turntable_sheet.md](../analysis/viz/turntable_sheet.md).

## What to look for
- **Wrong part depth** — a component pushed too far forward or back relative to
  the rest of the object.
- **Gross interpenetration** — articulated parts visibly cutting through one
  another in an observed state.
- **Geometry wrong from a hidden side** — anything the single matched view can't
  show (a caved-in back, a part only half-built). The `el+60` and `el-30` tiles
  exist for exactly this: a **caved-in top or an unbuilt base** is invisible in a
  level ring, and so was invisible in this view until the set gained them.

If the shape looks incoherent but you cannot tell which part is responsible,
render the `ids` view (which part owns a region) or `visibility` view (per-part
extent and what occludes what). Correct the joint origin/axis if its child follows
the wrong path. Add mechanism geometry only when source evidence requires it
(see AGENT_TASK.md rule 9).

## Run
```bash
harness/render.sh RUN_DIR/scene.py RUN_DIR/views \
    --views match,turntable --tracking CAPTURE/tracking --frames 000000.jpg,000080.jpg \
    --match-res <ref image> --samples 64
```

## The view set

Which directions get rendered is `--turntable-views`:

| set | views | what it is |
|---|---|---|
| `sphere8` (default) | 8 | the level ring **plus** a raised pair (`el+60`) and a dropped pair (`el-30`) — the top and underside are actually seen |
| `ring4` | 4 | the level ring only; half the renders per state, and blind above/below |

The set is **deterministic**: the same directions every pass, so this iteration's
`az180_el+20` is the same viewpoint as the last one's and "the back caved in
between iter03 and iter04" is a thing you can say. `--turntable-jitter SEED`
perturbs the directions when you deliberately want to look from somewhere new
(same seed → same set; the angles land in the filenames either way).

Framing flags: `--elevation` sets the **level ring's** elevation (the raised and
dropped views have their own absolute elevations, so raising the ring cannot push
them past the pole); `--azimuth` rotates the **whole set** about the up axis;
`--radius` / `--fov` / `--ortho` control distance and projection. The directions
and the filename grammar are defined in
[`core/turntable_views.py`](../core/turntable_views.py).

Cost scales as **states × views**: a run with 8 observed states renders 64 images
under `sphere8`. That is what the sheets are for; reach for `ring4` when you want
the cheap pass.

## Outputs (in the pass dir `render.sh` prints)
Each `render.sh` call writes into a fresh `RUN_DIR/views/<NNNN>/` pass dir and
PRINTS that dir; read this run's outputs from the printed dir. A view not rendered
this pass is absent from it (no stale leftovers).

- `turntable_<state>_az<AAA>_el<±EE>.png` — one per orbit view per unique
  articulation state. `<state>` is labelled by the first frame that exhibits it (a
  rigid object with no joints yields a single `turntable_rest_*`); `az`/`el` are the
  direction it was shot from, the elevation explicitly signed because that sign is
  the difference between looking **down** at the lid and **up** at the base.
- `turntable_sheets/` — the per-state sheets above (written by the shape pass, or by
  hand).

Glob these as `turntable_*_az*_el*.png`.

The object is oriented by the reference frame's pose (so it reads naturally
upright) with each state's joints applied; the turntable's job is to vet geometry,
not pose. This is a visual coherence gate, not a swept-clearance or self-collision
certification.
