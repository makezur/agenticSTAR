"""visual_budget.py — one guard against accidentally ordering thousands of panels.

`visuals: "all"` IS a supported order: an agent that wants every scored candidate
panelled can have it. The problem is that "all" is a property of the GRID, not of
the agent's intent — `yaw:-180,180,36 pitch:-60,60,12` is 432 renders, and the
agent typed a search, not a render budget. So the count is checked and a clearly
accidental one is refused with a message that names the waiver.

The threshold is 100: above any deliberate render set, below almost every
`visuals: "all"` expansion of a dense grid.

PURITY: stdlib only, no cv2/numpy/bpy — so `pool.manager` (which runs in the
`artscript` env with no bpy) and the analysis-env sheet builders share one guard
and one message.
"""

DEFAULT_BUDGET = 100

WAIVER_KEYS = ("waive_visual_budget",)

# The visuals LEVELS, in one place, so the default cannot drift between consumers.
VISUALS_LEVELS = ("auto", "all", "none")
DEFAULT_VISUALS = "auto"

# A named candidate SET (`--oapply 'flips'`) of at most this many candidates renders
# EVERY candidate, per frame, instead of binning — the whole point of a flip panel is
# that the agent LOOKS instead of trusting a score, and flip twins routinely tie on
# IoU. Above it, the set bins like any other sweep.
#
# It lives here, beside DEFAULT_BUDGET, because it is the same KIND of thing: a policy
# on how many images an order may produce. The docs (`rig/args.py`'s `--oapply` help,
# `views/sweeps/oapply.md`, `views/sweeps/oapply_all.md`) name this constant.
MAX_SET_RENDERS = 8


class VisualBudgetError(ValueError):
    """An order asked for more panels than the budget and did not waive it."""


def visuals_level(value):
    """Normalize a `visuals` setting to one of VISUALS_LEVELS.

    Anything unrecognized (including None and "") falls back to DEFAULT_VISUALS,
    because a typo must not silently mean "none" and lose the agent's panels."""
    level = str(value or DEFAULT_VISUALS).strip().lower()
    return level if level in VISUALS_LEVELS else DEFAULT_VISUALS


def check_budget(count, budget=0, waived=False, what="", detail=""):
    """Raise VisualBudgetError when `count` panels exceed the budget.

    `budget` <= 0 selects DEFAULT_BUDGET; pass an explicit value to tighten or
    loosen it. `waived` short-circuits the check (the agent has decided) but still
    reports what it is proceeding with, so a waived run is never silent about its
    size. Returns the resolved count so callers can log it.
    """
    limit = int(budget) if int(budget or 0) > 0 else DEFAULT_BUDGET
    count = int(count)
    if count <= limit:
        return count
    subject = f"{what}: " if what else ""
    if waived:
        print(f"[visual-budget] {subject}rendering {count} panels "
              f"(over the {limit} budget; waived)")
        return count
    raise VisualBudgetError(
        f"{subject}that order means {count} panels"
        + (f" ({detail})" if detail else "")
        + f", and the budget is {limit}. That is almost certainly not what you "
        "meant. Either narrow the grid, or drop "
        "dump_topk to panel the best candidate per ordered cell instead. To do it "
        "anyway pass waive_visual_budget: true on the order "
        "(--waive-visual-budget on the CLI).")
