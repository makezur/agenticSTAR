"""Adaptive search policies for the camera-frame sweep engine.

The engine owns Blender state and candidate scoring. This module owns only the
numerical optimizer: it receives a bounded vector objective and returns a trace.
Keeping that boundary explicit lets grid and adaptive searches share the exact
same Space.realize -> render -> metrics.score_of path.
"""

from dataclasses import dataclass, field

import numpy as np

from views.sweeps.lib import dof_space
from views.sweeps.lib._vendor.scipy_de import differential_evolution_best1bin
from views.sweeps.lib._vendor.scipy_powell import minimize_powell_bounded


class SearchStopped(Exception):
    """Internal control flow for a wall-clock deadline between evaluations."""


@dataclass
class SearchStage:
    kind: str
    n: int
    bounds: dict


@dataclass
class SearchOutcome:
    stages: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def parse_mutation(spec):
    """Parse SciPy's fixed F or ``min,max`` dithering interval."""
    values = [float(v.strip()) for v in str(spec).split(",") if v.strip()]
    if len(values) not in (1, 2):
        raise ValueError("--sweep-de-mutation must be F or min,max")
    if any(v < 0.0 or v >= 2.0 for v in values):
        raise ValueError("--sweep-de-mutation values must satisfy 0 <= F < 2")
    if len(values) == 2:
        if values[0] > values[1]:
            raise ValueError("--sweep-de-mutation min must not exceed max")
        return tuple(values)
    return values[0]


def _held_value(space, name):
    if space.kinds[name] == dof_space.POSE:
        return 0.0
    if name not in space.base_states:
        raise ValueError(
            f"DE: joint {name!r} has no base state to hold while other DOFs search")
    return float(space.base_states[name])


def vector_bounds(space, bounds):
    """Full-vector bounds and initial point, holding omitted DOFs at their no-op."""
    full_bounds = []
    x0 = []
    for name in space.dof_names:
        hold = _held_value(space, name)
        lo, hi = bounds.get(name, (hold, hold))
        lo, hi = float(lo), float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(f"DE: {name!r} bounds must be finite")
        if hi < lo:
            raise ValueError(f"DE: {name!r} lower bound {lo} exceeds upper {hi}")
        full_bounds.append((lo, hi))
        x0.append(min(max(hold, lo), hi))
    return full_bounds, np.asarray(x0, dtype=float)


def run_de(space, bounds, evaluate, should_stop=lambda: False, *,
           max_evals=1200, popsize=8, seed=17, mutation=(0.5, 1.0),
           recombination=0.8, polish="none", polish_max_evals=200):
    """Run bounded SciPy DE and optional Powell polish.

    ``evaluate(full_vector, stage_index) -> score`` is maximized. SciPy minimizes,
    so the adapter negates it. Duplicate vectors are cached across DE and polish;
    only unique evaluations reach Blender.
    """
    full_bounds, full_x0 = vector_bounds(space, bounds)
    active = [i for i, (lo, hi) in enumerate(full_bounds) if hi > lo]
    if not active:
        n = 0
        if not should_stop():
            evaluate(full_x0, 0)
            n = 1
        return SearchOutcome(
            stages=[SearchStage("de", n, dict(bounds))],
            metadata={"seed": int(seed), "active_dimensions": 0,
                      "max_evals": int(max_evals), "nfev": n,
                      "termination": ("fixed_bounds" if n else "timeout"),
                      "polish": "none"})

    active_bounds = [full_bounds[i] for i in active]
    active_x0 = full_x0[active]
    max_evals = max(5, int(max_evals))
    popsize = max(1, int(popsize))
    population_n = max(5, popsize * len(active))
    maxiter = max(0, max_evals // population_n - 1)
    cache = {}
    best = {"score": -np.inf, "x": full_x0.copy()}
    generation_best = []
    stage = [0]
    stage_counts = [0, 0]

    def full_vector(active_x):
        x = full_x0.copy()
        x[active] = np.asarray(active_x, dtype=float)
        return x

    def objective(active_x):
        x = full_vector(active_x)
        key = tuple(np.round(x, 12))
        if key in cache:
            return -cache[key]
        if should_stop():
            raise SearchStopped()
        score = float(evaluate(x, stage[0]))
        cache[key] = score
        stage_counts[stage[0]] += 1
        if score > best["score"]:
            best["score"], best["x"] = score, x.copy()
        return -score

    def callback(intermediate_result):
        generation_best.append(round(float(-intermediate_result.fun), 8))
        return should_stop()

    de_result = None
    termination = "completed"
    try:
        de_result = differential_evolution_best1bin(
            objective, active_bounds, maxiter=maxiter,
            popsize=popsize, mutation=mutation, recombination=float(recombination),
            seed=int(seed), callback=callback, x0=active_x0,
            maxfun=max_evals)
        if should_stop():
            termination = "timeout"
        else:
            termination = "converged" if de_result.success else "max_evals"
    except SearchStopped:
        termination = "timeout"

    de_best_score = None if not np.isfinite(best["score"]) else float(best["score"])
    polish_result = None
    if (polish == "powell" and de_best_score is not None
            and int(polish_max_evals) > 0 and not should_stop()):
        stage[0] = 1
        try:
            polish_result = minimize_powell_bounded(
                objective, best["x"][active], active_bounds,
                maxfev=int(polish_max_evals), xtol=2e-3, ftol=1e-6)
        except SearchStopped:
            termination = "timeout"

    stages = [SearchStage("de", stage_counts[0], dict(bounds))]
    if stage_counts[1]:
        stages.append(SearchStage("polish", stage_counts[1], dict(bounds)))
    return SearchOutcome(
        stages=stages,
        metadata={
            "strategy": "best1bin",
            "seed": int(seed),
            "active_dimensions": len(active),
            "max_evals": max_evals,
            "population_multiplier": popsize,
            "population_size": population_n,
            "max_generations": maxiter,
            "mutation": ([float(mutation[0]), float(mutation[1])]
                         if isinstance(mutation, tuple) else float(mutation)),
            "recombination": float(recombination),
            "generation_best": generation_best,
            "de_best": de_best_score,
            "de_nfev": (None if de_result is None else int(de_result.nfev)),
            "polish": polish,
            "polish_nfev": (None if polish_result is None
                            else int(polish_result.nfev)),
            "nfev": len(cache),
            "termination": termination,
        })
