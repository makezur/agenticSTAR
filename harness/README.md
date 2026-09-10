# `harness/` — design principles

The harness is the fixed machinery around the agent's work. Read this before
changing anything here; it states the invariants future runs and edits must keep.

## The one big idea: build canonical once, pose + articulate per frame

The agent solves three **decoupled** problems, never one tangled one:
- **Shape** — `build()` creates named, watertight parts in a **canonical frame**
  (+Z up, centered at origin, longest dim ≈ 1), camera-unaware, in their rest
  configuration; the **union of parts across all frames**.
- **Pose** — per frame, `FRAMES[name]["pose"]` (R, t) + a **shared** `SCALE`
  states how the canonical object sits in front of that frame's camera. Frame
  cameras are **known** (from the capture's tracking), so non-reference poses are auto-seeded.
- **Articulation** — `JOINTS` (shared definitions) + `FRAMES[name]["joints"]`
  (per-frame DOF states) move a part relative to the body between frames.

Everything else follows from keeping these separate. The exported GLB is the
**canonical rest pose**; per-frame poses + joint states live only in `pose.json`.
That split is the URDF/articulation handoff — don't bake pose or state into
vertices. `SCALE` is shared across frames (physical size is frame-invariant).

## Ownership boundary

- **Agent owns** `scene.py`: geometry, materials, `SCALE`, `REFERENCE_FRAME`,
  `JOINTS`, and `FRAMES` (per-frame poses + joint states). Nothing else.
- **Harness owns** the cameras, lighting, render/GPU settings, pose *application*,
  forward kinematics, and export. The agent never edits anything under `harness/`.

Every frame renders from the **fixed camera 0** (identity extrinsic); per-frame
intrinsics come from the capture's tracking (or a placeholder). The agent matches each photo by
posing/articulating the object, **not** by moving the camera — the harness moves
the object into each frame's known camera. A silhouette mismatch is therefore
always shape, per-frame pose, or joint state — never viewpoint.

## Code layout

- `render_wrapper.py` — thin dispatcher (build → per-frame pose+FK → run views →
  pose.json → export). Keep it thin; logic belongs in `rig/` or `views/`.
- `rig/` — shared Blender-side core (agent never edits). See [rig/README.md](rig/README.md).
- `views/` — one module + one `.md` per render view. See [views/README.md](views/README.md).
- `analysis/` — out-of-Blender CLIs (artscript env). See [analysis/README.md](analysis/README.md).
- `utils/` — orchestration CLIs (artscript env): `shape_pass.sh` runs the full
  shape diagnostic pass; `composite_pass.sh` renders match views and review
  composites after a pose-windows apply.
- `multiagent/` — the parallel refinement round (artscript env): `windows.py`
  (per-frame refinement via pose-refiner subagents, each ordering its own sweeps)
  and `pool_session.py` (the locked shared-pool lifecycle a round runs under).
  Temporal residual reports come from `analysis/temporal/`. See
  [utils/shape_pass.md](utils/shape_pass.md),
  [multiagent/windows.md](multiagent/windows.md), and
  [analysis/temporal/report.md](analysis/temporal/report.md).
- `pool/` — the resident Blender **render pool**: build once via
  `render_wrapper.py --serve`, render many. `manager.py` (the pool manager,
  `python -m pool.manager`), `serve.py` (the in-Blender worker loop),
  `client.py` / `session.py` / `locks.py` (spool client, process lifecycle,
  locking). Wired into `shape_pass.sh` via `--pool`. See [pool/README.md](pool/README.md).

**Two runtimes.** `render.sh` runs inside Blender's bundled Python; everything in
`analysis/` runs via `micromamba run -n artscript`. Don't mix deps across them.

## Principles for changes

1. **Docs live with the code, once.** Each agent-facing tool has a co-located
   `.md` that is its single source of truth. `argparse --help` is ground truth
   for flag semantics. `AGENT_TASK.md`/top-level `README.md` only *index* tools —
   never re-document them (that drift is what this structure exists to kill).
2. **Views are non-destructive.** Every view must snapshot and restore whatever it
   touches (materials, color management, intrinsics, hide flags, world matrices),
   so the per-frame `match`/`depth`, the per-state `turntable`, and the GLB are
   unaffected. The render loop poses parts per frame FROM the captured canonical
   matrices and restores them after each frame; export runs on canonical. Verify
   this when adding a view.
3. **Keep the silhouette exact.** Renders use a transparent film so the alpha
   channel is a true silhouette (the IoU depends on it). Don't add opaque
   backgrounds or post-effects that leak into alpha.
4. **Adding a tool:** a render view = a module exposing `render_view(ctx)` +
   registered in `views/__init__.py` + a `<name>.md`; an analysis step = a CLI in
   `analysis/` + a `.md`. Add its row to the relevant folder README and the
   `AGENT_TASK.md` index.
5. **Parts stay separate and manifold** for downstream articulation — don't
   union things that will move relative to each other.
