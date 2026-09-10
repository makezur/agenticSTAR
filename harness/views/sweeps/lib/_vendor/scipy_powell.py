"""Contained bounded Powell path from SciPy 1.18.

Derived from scipy/optimize/_optimize.py at SciPy v1.18.0:
https://github.com/scipy/scipy/blob/v1.18.0/scipy/optimize/_optimize.py

Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
Distributed under the BSD-3-Clause license in ``SCIPY_LICENSE.txt``.

Only the finite-bounds path used to polish sweep results is included.
"""

from dataclasses import dataclass

import numpy as np


class _MaxFuncCallError(RuntimeError):
    pass


@dataclass
class PowellResult:
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    success: bool
    message: str


@dataclass
class _ScalarResult:
    x: float
    fun: float
    nfev: int


def _minimize_scalar_bounded(func, bounds, xatol=1e-5, maxiter=500):
    """SciPy's bounded scalar Brent minimizer."""
    x1, x2 = bounds
    if not np.isfinite(x1) or not np.isfinite(x2):
        raise ValueError("optimization bounds must be finite")
    if x1 > x2:
        raise ValueError("the lower bound exceeds the upper bound")

    sqrt_eps = np.sqrt(2.2e-16)
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))
    a, b = x1, x2
    fulc = a + golden_mean * (b - a)
    nfc = xf = fulc
    rat = e = 0.0
    fx = func(xf)
    num = 1
    fu = np.inf
    ffulc = fnfc = fx
    xm = 0.5 * (a + b)
    tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
    tol2 = 2.0 * tol1

    while np.abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        golden = True
        if np.abs(e) > tol1:
            golden = False
            r = (xf - nfc) * (fx - ffulc)
            q = (xf - fulc) * (fx - fnfc)
            p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = np.abs(q)
            r = e
            e = rat
            if (np.abs(p) < np.abs(0.5 * q * r)
                    and p > q * (a - xf) and p < q * (b - xf)):
                rat = p / q
                x = xf + rat
                if (x - a) < tol2 or (b - x) < tol2:
                    sign = np.sign(xm - xf) + (xm == xf)
                    rat = tol1 * sign
            else:
                golden = True

        if golden:
            e = a - xf if xf >= xm else b - xf
            rat = golden_mean * e

        sign = np.sign(rat) + (rat == 0)
        x = xf + sign * np.maximum(np.abs(rat), tol1)
        fu = func(x)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc, ffulc = nfc, fnfc
            nfc, fnfc = xf, fx
            xf, fx = x, fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if fu <= fnfc or nfc == xf:
                fulc, ffulc = nfc, fnfc
                nfc, fnfc = x, fu
            elif fu <= ffulc or fulc == xf or fulc == nfc:
                fulc, ffulc = x, fu

        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * np.abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= maxiter:
            break

    return _ScalarResult(float(xf), float(fx), num)


def _line_for_search(x0, direction, lower_bound, upper_bound):
    nonzero, = direction.nonzero()
    lower = lower_bound[nonzero]
    upper = upper_bound[nonzero]
    position = x0[nonzero]
    delta = direction[nonzero]
    low = (lower - position) / delta
    high = (upper - position) / delta
    positive = delta > 0
    line_min = np.max(np.where(positive, low, high))
    line_max = np.min(np.where(positive, high, low))
    return (line_min, line_max) if line_max >= line_min else (0.0, 0.0)


def _linesearch_powell(
        func, position, direction, *, tol, lower_bound, upper_bound, fval):
    if not np.any(direction):
        return fval, position, direction

    def line_func(alpha):
        return func(position + alpha * direction)

    bound = _line_for_search(
        position, direction, lower_bound, upper_bound)
    result = _minimize_scalar_bounded(
        line_func, bound, xatol=tol / 100.0)
    moved = result.x * direction
    return result.fun, position + moved, moved


def minimize_powell_bounded(
        func, x0, bounds, *, maxfev=200, xtol=2e-3, ftol=1e-6):
    """Minimize with SciPy 1.18's finite-bounds Powell path."""
    x = np.asarray(x0, dtype=float).flatten()
    limits = np.asarray(bounds, dtype=float)
    if limits.shape != (len(x), 2) or not np.all(np.isfinite(limits)):
        raise ValueError("bounds must be finite (min, max) pairs matching x0")
    lower_bound, upper_bound = limits[:, 0], limits[:, 1]
    if np.any(lower_bound > upper_bound):
        raise ValueError("a lower bound exceeds its upper bound")
    if np.any(lower_bound > x) or np.any(x > upper_bound):
        raise ValueError("x0 must lie within the bounds")

    calls = [0]

    def counted(value):
        if calls[0] >= int(maxfev):
            raise _MaxFuncCallError()
        result = float(func(value))
        calls[0] += 1
        return result

    directions = np.eye(len(x), dtype=float)
    fval = counted(x)
    previous_x = x.copy()
    iterations = 0
    converged = False

    while True:
        try:
            start_value = fval
            biggest_index = 0
            biggest_drop = 0.0
            for index in range(len(x)):
                direction = directions[index]
                before = fval
                fval, x, direction = _linesearch_powell(
                    counted, x, direction, tol=xtol * 100.0,
                    lower_bound=lower_bound, upper_bound=upper_bound,
                    fval=fval)
                if before - fval > biggest_drop:
                    biggest_drop = before - fval
                    biggest_index = index

            iterations += 1
            threshold = ftol * (
                np.abs(start_value) + np.abs(fval)) + 1e-20
            if 2.0 * (start_value - fval) <= threshold:
                converged = True
                break
            if calls[0] >= int(maxfev):
                break
            if np.isnan(start_value) and np.isnan(fval):
                break

            direction = x - previous_x
            previous_x = x.copy()
            _, line_max = _line_for_search(
                x, direction, lower_bound, upper_bound)
            extrapolated = x + min(line_max, 1.0) * direction
            extrapolated_value = counted(extrapolated)

            if start_value > extrapolated_value:
                test = 2.0 * (
                    start_value + extrapolated_value - 2.0 * fval)
                test *= (start_value - fval - biggest_drop) ** 2
                test -= biggest_drop * (
                    start_value - extrapolated_value) ** 2
                if test < 0.0:
                    fval, x, direction = _linesearch_powell(
                        counted, x, direction, tol=xtol * 100.0,
                        lower_bound=lower_bound, upper_bound=upper_bound,
                        fval=fval)
                    if np.any(direction):
                        directions[biggest_index] = directions[-1]
                        directions[-1] = direction
        except _MaxFuncCallError:
            break

    if converged:
        message = "optimization converged"
    elif calls[0] >= int(maxfev):
        message = "maximum function evaluations reached"
    else:
        message = "optimization stopped"
    return PowellResult(
        x=x.copy(), fun=float(fval), nfev=calls[0], nit=iterations,
        success=converged, message=message)
