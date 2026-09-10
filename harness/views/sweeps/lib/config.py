"""config.py — the resolved knobs the thin views hand the engines.

Every sweep-family verb is a thin adapter: it reads `ctx.args` (whose flags are
declared, and documented, in `rig/args.py`) and builds ONE of these, then calls its
engine. Keeping the configs here, `bpy`-free and beside `dof_space.Space`, means:

  * the two engines can share the SCORING knobs they genuinely have in common
    (`ScoringConfig` below) instead of restating them — 11 of them, and every one
    reaches the same `runtime.ScoringSession` / `metrics.score_of` code;
  * a config can be built and asserted in the analysis env, with no Blender;
  * they are dataclasses, so a mistyped field raises instead of silently falling
    back to a default. Plain attribute access is the safety feature.

Do NOT put resolution logic here. These are values; the engines interpret them
(e.g. `""` space -> inferred from the named DOFs, which needs the scene's joints).

Every field carries a default, because a subclass field cannot follow a defaulted
base field otherwise. The mask fields are still REQUIRED in the sense that matters —
both engines refuse an empty one with a named skip message before any render — so
the guard lives where it can say something useful, not in the constructor.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

# the two selection modes of the canonical engine (see SharedConfig.mode).
SHARED = "shared"
PER_FRAME = "per_frame"


@dataclass
class ScoringConfig:
    """What every sweep-family verb needs to SCORE candidates, whichever frame it
    searches in. Both engines read exactly these fields the same way, so a change
    here changes both — which is the point.

    Masks are NOT here: the camera-frame engine scores one frame from a single mask
    path, the canonical engine scores N frames from a directory. That difference is
    real, so each subclass declares its own.
    """

    # scoring inputs
    hand_dilate: float = 0.0        # dilate the hand mask before it occludes (px, at
                                    # the scoring resolution)
    quality: float = 1.0            # scoring resolution + default grid density knob
    # weight of the observed-depth penalty in `combined` (0 -> pure silhouette
    # IoU). None = AUTO: the per-run depth_config.json's `weight` (the backend's
    # default), else 0.1 (runtime.resolve_depth_weight). A caller that wants
    # depth to bite (e.g. a pool order asking for 0.5) states it per-order.
    depth_weight: Optional[float] = None
    timeout: float = 0.0            # wall-clock budget in seconds; <=0 -> none

    # candidate generation
    ranges: str = ""                # the verb's own range grammar (see planner)
    angle_preset: str = ""          # per-axis size preset for the default box
    refine: int = 1                 # coarse->fine passes (0 for imperative verbs)
    refine_shrink: float = 0.5      # each refine pass shrinks the window by this

    # how many candidates become IMAGES: "auto" honours `dump_topk` (with each
    # engine's own floor — at least the winner), "all" panels every scored
    # candidate, "none" scores only. "all" is a supported order but is checked
    # against core/visual_budget, because "all" is a property of the GRID (and, for
    # the canonical engine, of the frame count), not of the agent's intent.
    visuals: str = "auto"
    waive_visual_budget: bool = False
    dump_topk: int = 0              # how many non-winner candidates to re-render

    # output identity. The imperative verbs run the same engine as their searching
    # sibling but must not collide with a co-requested search, so each names its own
    # outputs and labels its own logs.
    out_stem: str = "sweep"         # <stem>_best.png / <stem>.json / <stem>.txt
    view: str = "sweep"             # log prefix + manifest "view" label


@dataclass
class SweepConfig(ScoringConfig):
    """Resolved knobs for the CAMERA-FRAME engine (`engine.run`): the `sweep`
    search and the imperative `apply`. Built by views/sweeps/sweep.py and
    views/apply.py from ctx.args."""

    mask: str = ""                  # required: the object mask for ctx.frame
    hand_mask: str = ""
    space: str = ""                 # "" (infer from the named DOFs) | pose | joint
                                    #    | both
    # 0 = strategy default: six restart winners for DE, ten score-ranked entries
    # for grid/candidates. A positive value is literal.
    topk: int = 0
    # WHERE THE SEARCH STARTS — the placement its offsets are measured from, i.e. the
    # candidate at offset (0,0,0). "" = the posed frame's own placement
    # (engine._resolve_pose_start). Placement ONLY: it carries no joints (that is
    # `start_shift`) and no scale (the engine applies the object's shared SCALE).
    pose_start: str = ""            # --sweep-pose-start
    # a directional ORDER (--apply grammar) composed onto `pose_start` BEFORE the
    # search, so the sweep hunts a window AROUND the shifted pose (and carries
    # joints, unlike `pose_start`). "" = no shift. See engine._apply_start_shift.
    start_shift: str = ""           # --sweep-start-shift
    candidates: str = ""            # explicit candidate list -> bypasses search
    grid_slice: str = ""            # "i/k": shard the grid across parallel runs
    joints: str = ""                # selector for the empty-spec joint default box
    strategy: str = "de"            # de | grid
    de_max_evals: int = 3600        # total across all top-K DE restarts
    de_popsize: int = 8             # SciPy multiplier: population ~= value * dims
    de_seed: int = 17
    de_mutation: str = "0.5,1.0"    # fixed F or dithering interval "min,max"
    de_recombination: float = 0.8   # crossover probability CR
    polish: str = "none"            # none | powell
    polish_max_evals: int = 200


@dataclass
class SharedConfig(ScoringConfig):
    """Resolved knobs for the CANONICAL engine (`shared_engine.run`): `osweep`,
    `oapply`, `oapply_all`. Built by views/sweeps/osweep.py, views/oapply.py, and
    views/oapply_all.py from ctx.args."""

    masks_dir: str = ""             # required: per-frame object masks
    hand_masks_dir: str = ""
    # SHARED    — ONE rotation for every frame, ranked by MEAN gate IoU
    #             (oapply_all: a build-orientation error, one decision).
    # PER_FRAME — one rotation PER frame, each its own argmax. Semantics are exactly
    #             N independent sweep/apply runs that happen to share a Blender
    #             process and a candidate grid: nothing aggregated, nothing
    #             inherited. Only the SELECTION step and the report differ — the
    #             render loop is the same frame x candidate matrix.
    mode: str = SHARED
    # --opreset: 'axis:turnset|sizeband|anglelist' (see planner.parse_opreset).
    # The SEARCH verb turns it into the candidate grid; the imperative verbs expand
    # it to a labelled candidate SET before building their config, so by the time it
    # reaches here it only matters to grid mode.
    preset: str = ""
    # explicit candidate SET [(label, (rx,ry,rz)), ...] — bypasses the grid entirely
    # (oapply's flip panels / custom sets, from planner.parse_canon_candidates).
    # Discrete named rotations, so refine never applies to them.
    candidates: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None
