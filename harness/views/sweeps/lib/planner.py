"""planner.py — the FACADE over the sweep family's two candidate grammars.

The sweep family searches in two different frames, and the two grammars share
nothing but arithmetic: disjoint DOF names, disjoint spec syntax, disjoint presets,
disjoint engines. So they are two modules —

  * `planner_pose`  — CAMERA-frame: roll/yaw/pitch/dpx/dpy/tz increments
                       left-multiplied onto ONE frame's pose, plus absolute JOINT
                       states, over a Cartesian `grid`. Used by `engine`
                       (`sweep`, `apply`).
  * `planner_canon` — OBJECT-frame: an rx/ry/rz rotation VECTOR in the object's own
                       canonical frame, right-multiplied onto EVERY frame's pose.
                       Used by `shared_engine` (`osweep`, `oapply`, `oapply_all`).
  * `planner_num`   — the little both share: the named angle-preset SIZES,
                       `steps_for_quality`, `_linspace`.

— and this module re-exports both under the one name every caller already imports.
That is the whole job: `planner.parse_ranges` and `planner.parse_canon_ranges`
still resolve, so the split touched no engine, view, or test.

Prefer importing the specific module in NEW code
(`from views.sweeps.lib import planner_canon`), which says which frame you are in. Keep this facade for the
existing call sites, and because "the planner" is how the docs name the concept.

PURITY: pure numpy + stdlib + the pure `dof_space`. No `bpy`.
"""

# flake8: noqa: F401  (a facade: every import here exists to be re-exported)

# the removed-DOF (`sigma`) teaching error lives with the DOF vocabulary itself.
from views.sweeps.lib.dof_space import removed_dof_error
from views.sweeps.lib.planner_num import (ANGLE_AXES, ANGLE_DIRECTIONS,
                                       ANGLE_PRESET_HALF, DEFAULT_ANGLE_HALF,
                                       DEFAULT_ANGLE_PRESET, steps_for_quality)
from views.sweeps.lib.planner_pose import (DEFAULT_PRISMATIC, DEFAULT_REVOLUTE,
                                        POSE_COUPLED_DEFAULT, POSE_DEFAULT_HALF,
                                        apply_ranges_spec, default_ranges, grid,
                                        grid_budget_estimate, parse_angle_presets,
                                        parse_apply_pairs, parse_bounds, parse_candidates,
                                        parse_ranges, refine_ranges)
from views.sweeps.lib.planner_canon import (CANON_DOFS, FLIP_AXIS_DOF,
                                         FLIPS_PANEL_SPEC, SO3_MAX, SO3_SEED,
                                         TURN_SETS, CanonPlan, So3Plan,
                                         canon_cross_hint, canon_grid,
                                         canon_report_ranges, canon_searched_dofs,
                                         default_canon_ranges,
                                         imperative_candidates,
                                         merge_canon_candidates,
                                         opreset_candidates,
                                         parse_canon_angle_presets,
                                         parse_canon_candidates,
                                         parse_canon_ranges, parse_opreset,
                                         refine_canon_ranges,
                                         resolve_canon_search,
                                         so3_cell_half_deg, so3_samples)
