# `shapes/` — shared build-time helpers for `scene.py`

The boilerplate mesh/material helpers an agent-authored `scene.py` calls inside
`build()` to make canonical geometry. Unlike `rig/`/`views/` (fixed rig the agent
never touches), this is a small **API the agent imports** — one canonical copy, so
don't redefine these in your scene:

```python
from shapes import box, convex_hull, set_color, apply_boolean
```

| helper | does |
|---|---|
| `box(name, center, size)` | axis-aligned cube of **FULL** extent `size = (sx, sy, sz)`, centered at `center`. Sizes to full extent — never `size/2`. |
| `convex_hull(name, points, recalc=True)` | one **CONVEX** solid = the convex hull of the `(x,y,z)` `points`; inherently watertight. Carve concavities/holes **afterward** with `apply_boolean` (interior points are discarded). Needs ≥4 non-coplanar/non-collinear points or it raises. |
| `set_color(obj, rgb, roughness=0.5, metallic=0.0)` | give `obj` a Principled BSDF material (`rgb` in 0..1). |
| `apply_boolean(target, cutter, operation="DIFFERENCE")` | apply a boolean modifier (EXACT solver) and delete the cutter. |

Everything is authored in the **CANONICAL frame**: +Z up, centered at the origin,
longest dimension ≈ 1 unit. `convex_hull`/`box` link their object into the active
scene collection, so `build_from()` (`../rig/scene.py`) collects it into the
`parts` collection like any primitive.

## Building geometry — primitives, hulls, booleans (the hints)

You build from **primitives** (cube / cylinder / sphere / cone / torus, or a
`convex_hull` of points) plus transforms and modifiers: boolean
(`solver="EXACT"`), bevel, solidify, mirror. Shapes built this way are watertight
and decompose cleanly. Start simple and subdivide only if the shape isn't
satisfactory; it's fine to stitch many primitives into one complex shape.

- **Cube extent.** `primitive_cube_add(size=1.0)` already spans ±0.5 (full extent
  1.0), so set `obj.scale` to the **full** dimension you want, never `size/2`. The
  `box()` helper does this for you.
- **`convex_hull` is CONVEX-ONLY.** You cannot dent, notch, or hollow a hull by
  adding interior points (they are discarded). For a convex shape the fixed
  primitives can't express — wedge, prism, tapered/faceted block, trapezoidal
  body — hull a point set; carve any concavity/hole **afterward** with a boolean
  **DIFFERENCE** cutter (or use `torus` for a ring).
- **A `DIFFERENCE` stays watertight only if the cutter fully protrudes** through
  every surface it crosses — extend it a hair past both sides, leave no coplanar
  face.
- **Never boolean-UNION parts that articulate** relative to each other (a door vs
  its body — see the articulation rule in the task): keep them separate objects
  so forward kinematics can move them independently. Their meshes may have
  designed clearance and need not touch. Small intentional intersections remain
  useful for rigid attachments, such as seating a fixed handle into a door. Only
  union primitives that truly form one rigid piece.

## Boundaries a refactor must keep
- This package imports `bpy`/`bmesh`, so it lives on the **Blender side** next to
  `rig/`/`views/` — never under `core/` (numpy-only).
- `harness/` is on `sys.path` at scene-exec time (`render_wrapper.py` inserts it),
  so scenes run via `runpy.run_path` can `from shapes import ...` directly. Keep
  this package in `harness/`.
