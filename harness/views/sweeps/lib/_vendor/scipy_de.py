"""Contained SciPy 1.18 ``best1bin`` differential-evolution kernel.

Derived from scipy/optimize/_differentialevolution.py at SciPy v1.18.0:
https://github.com/scipy/scipy/blob/v1.18.0/scipy/optimize/_differentialevolution.py

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
Distributed under the BSD-3-Clause license in ``SCIPY_LICENSE.txt``.

This is deliberately not a general replacement for
``scipy.optimize.differential_evolution``. It contains only the path used by
the Blender sweep engine: finite box bounds, ``best1bin``, Latin-hypercube
initialization, immediate updates, one scalar worker, optional dithering, and
an initial point. The numerical operations and RNG call order follow SciPy.

``maxfun`` is the one local extension. SciPy's public function budgets complete
generations, but a render is expensive enough that this caller needs a hard
objective-call ceiling, including a partial initial population.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class DifferentialEvolutionResult:
    """Result fields consumed by the sweep policy and parity tests."""

    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    success: bool
    message: str
    population: np.ndarray
    population_energies: np.ndarray


def differential_evolution_best1bin(
        func, bounds, *, maxiter=1000, popsize=15, tol=0.01,
        mutation=(0.5, 1.0), recombination=0.7, seed=None, callback=None,
        x0=None, atol=0.0, maxfun=np.inf):
    """Minimize ``func`` using SciPy 1.18's serial, immediate ``best1bin`` path."""
    limits = np.asarray(bounds, dtype=float).T
    if (limits.ndim != 2 or limits.shape[0] != 2
            or not np.all(np.isfinite(limits))):
        raise ValueError(
            "bounds should contain finite real-valued (min, max) pairs")
    if np.any(limits[0] > limits[1]):
        raise ValueError("a lower bound exceeds its upper bound")

    mutation_values = np.asarray(mutation, dtype=float)
    if (mutation_values.ndim > 1 or mutation_values.size not in (1, 2)
            or not np.all(np.isfinite(mutation_values))
            or np.any(mutation_values < 0.0)
            or np.any(mutation_values >= 2.0)):
        raise ValueError("mutation must be F or (min, max), with 0 <= F < 2")
    dither = None
    scale = float(mutation_values.flat[0])
    if mutation_values.size == 2:
        dither = sorted(float(v) for v in mutation_values)

    parameter_count = limits.shape[1]
    if parameter_count < 1:
        raise ValueError("at least one bound is required")
    if not 0.0 <= float(recombination) <= 1.0:
        raise ValueError("recombination must be in [0, 1]")

    rng = np.random.RandomState(seed)
    equal_bounds = limits[0] == limits[1]
    population_n = max(
        5, int(popsize) * max(1, parameter_count - np.count_nonzero(equal_bounds)))
    population_shape = (population_n, parameter_count)

    # SciPy's init_population_lhs, including the per-column permutation order.
    segment = 1.0 / population_n
    samples = (
        segment * rng.uniform(size=population_shape)
        + np.linspace(0.0, 1.0, population_n, endpoint=False)[:, np.newaxis]
    )
    population = np.zeros_like(samples)
    for column in range(parameter_count):
        order = rng.permutation(range(population_n))
        population[:, column] = samples[order, column]

    scale_arg1 = 0.5 * (limits[0] + limits[1])
    scale_arg2 = np.abs(limits[0] - limits[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        reciprocal_scale = 1.0 / scale_arg2
        reciprocal_scale[~np.isfinite(reciprocal_scale)] = 0.0

    def scale_parameters(unit_values):
        return scale_arg1 + (unit_values - 0.5) * scale_arg2

    def unscale_parameters(parameters):
        return (parameters - scale_arg1) * reciprocal_scale + 0.5

    if x0 is not None:
        x0_scaled = unscale_parameters(np.asarray(x0, dtype=float))
        if x0_scaled.shape != (parameter_count,):
            raise ValueError("x0 must have one entry per bound")
        if ((x0_scaled > 1.0) | (x0_scaled < 0.0)).any():
            raise ValueError("some entries in x0 lie outside the bounds")
        population[0] = x0_scaled

    population_energies = np.full(population_n, np.inf)
    random_population_index = np.arange(population_n)
    nfev = 0

    def promote_lowest_energy():
        best_idx = int(np.argmin(population_energies))
        population_energies[[0, best_idx]] = population_energies[[best_idx, 0]]
        population[[0, best_idx], :] = population[[best_idx, 0], :]

    # SciPy evaluates the complete initial population. The local maxfun extension
    # truncates it because each function call is a Blender render.
    remaining = population_n if np.isinf(maxfun) else max(0, int(maxfun - nfev))
    initial_n = min(population_n, remaining)
    for candidate in range(initial_n):
        population_energies[candidate] = float(
            func(scale_parameters(population[candidate])))
        nfev += 1
    promote_lowest_energy()

    def result(nit, success, message):
        return DifferentialEvolutionResult(
            x=scale_parameters(population[0]).copy(),
            fun=float(population_energies[0]),
            nfev=nfev,
            nit=nit,
            success=success,
            message=message,
            population=scale_parameters(population).copy(),
            population_energies=population_energies.copy(),
        )

    if nfev >= maxfun:
        return result(0, False, "maximum function evaluations reached")

    nit = 0
    for nit in range(1, int(maxiter) + 1):
        if dither is not None:
            scale = rng.uniform(dither[0], dither[1])

        for candidate in range(population_n):
            if nfev >= maxfun:
                return result(nit - 1, False,
                              "maximum function evaluations reached")

            # SciPy's _mutate(candidate) for best1bin.
            fill_point = rng.randint(parameter_count)
            rng.shuffle(random_population_index)
            selected = random_population_index[:6]
            selected = selected[selected != candidate][:5]
            r0, r1 = selected[:2]
            mutant = population[0] + scale * (
                population[r0] - population[r1])

            crossovers = rng.uniform(size=parameter_count)
            crossovers = crossovers < float(recombination)
            crossovers[fill_point] = True
            trial = np.where(crossovers, mutant, population[candidate])

            out_of_bounds = np.bitwise_or(trial > 1.0, trial < 0.0)
            if count := np.count_nonzero(out_of_bounds):
                trial[out_of_bounds] = rng.uniform(size=count)

            energy = float(func(scale_parameters(trial)))
            nfev += 1
            if energy <= population_energies[candidate]:
                population[candidate] = trial
                population_energies[candidate] = energy
                if energy <= population_energies[0]:
                    promote_lowest_energy()

        if callback is not None:
            intermediate = result(nit, True, "in progress")
            if callback(intermediate):
                return result(nit, False, "callback function requested stop early")

        if not np.any(np.isinf(population_energies)):
            spread = np.std(population_energies)
            threshold = float(atol) + float(tol) * abs(
                np.mean(population_energies))
            if spread <= threshold:
                return result(nit, True, "optimization converged")

    # SciPy reports maxiter=0 as a successful initial-population evaluation.
    if int(maxiter) == 0:
        return result(0, True, "optimization completed")
    return result(nit, False, "maximum iterations reached")
