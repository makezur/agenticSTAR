"""views.sweeps.lib — the machinery the five sweep verbs share.

The verbs live one level up (`views/sweeps/sweep.py` and friends) and are thin by
construction: each resolves `ctx.args` into a config and calls an engine. Everything
those two lines stand on is here, split into blocks so each is small and testable:

  * `config`        — the resolved knobs each thin view hands its engine:
                       `SweepConfig` / `SharedConfig`, both dataclasses over one
                       shared `ScoringConfig` base (pure).
  * `dof_space`     — the umbrella: a DOF vector -> a posed candidate (pure).
  * `planner`       — candidate generation, one FACADE over two grammars that
                       share nothing but arithmetic (pure):
                         `planner_pose`  camera-frame ranges + `grid`,
                         `planner_canon` canonical rx/ry/rz ranges, turn sets,
                                          presets, and named candidate sets,
                         `planner_num`   the angle-preset sizes + `_linspace`.
                       Callers import `planner`; new code can name the half.
  * `metrics`       — score_of, landscape/plateau stats, the 2-D score field, the
                       panel plan, and the camera-frame report writer (pure).
  * `engine`        — the camera-frame engine: the Blender render/score loop,
                       depth supervision, winner re-render, report assembly.
                       Drives `sweep` and `apply`.
  * `shared_engine` — the canonical engine, in two modes: SHARED (one rotation
                       fitted across all frames) and PER_FRAME (one per frame).
                       Drives `osweep`, `oapply`, and `oapply_all`.
  * `shared_report` — the canonical engine's two report writers (pure).
  * `shared_placement` — the deg->rad / rounding / right-reorient seam that the
                       canonical engine and its report MUST agree on (pure).
  * `runtime`       — Blender-side runtime glue (Progress, DepthSupervision, the
                       scoring-resolution helpers), shared by both engines.

PURITY: everything above except `engine`, `shared_engine`, and `runtime` is
`bpy`-free and imports in the analysis env, so a stray `import bpy` there is caught
outside Blender rather than surfacing as a crash inside it.
"""
