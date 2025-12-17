"""
integrators.py

Numerical ODE integrators used in the project.

This module contains ONLY numerical methods.
It must not depend on any physical model (Yaman, FWM, optics, etc.).
"""

from __future__ import annotations

import numpy as np
from typing import Callable


# Type alias for RHS function
# f(z, y, params) -> dy/dz
RHSFunction = Callable[[float, np.ndarray, object], np.ndarray]


# ---------------------------------------------------------------------
# Single-step Runge–Kutta 4th order
# ---------------------------------------------------------------------

def rk4_step(
    f: RHSFunction,
    z: float,
    y: np.ndarray,
    dz: float,
    params: object,
) -> np.ndarray:
    """
    Perform a single RK4 step.

    Parameters
    ----------
    f : callable
        Right-hand side function f(z, y, params).
    z : float
        Current integration coordinate.
    y : np.ndarray
        Current state vector.
    dz : float
        Step size.
    params : object
        Arbitrary parameter container passed to f.

    Returns
    -------
    y_next : np.ndarray
        State vector after one RK4 step.
    """

    k1 = f(z, y, params)
    k2 = f(z + 0.5 * dz, y + 0.5 * dz * k1, params)
    k3 = f(z + 0.5 * dz, y + 0.5 * dz * k2, params)
    k4 = f(z + dz, y + dz * k3, params)

    y_next = y + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return y_next


# ---------------------------------------------------------------------
# Fixed-step integration over a predefined z-grid
# ---------------------------------------------------------------------

def integrate_fixed_step(
    f: RHSFunction,
    z_grid: np.ndarray,
    y0: np.ndarray,
    params: object,
    *,
    save_every: int = 1,
    check_nan: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE system using fixed-step RK4 over a given z-grid.

    Parameters
    ----------
    f : callable
        RHS function f(z, y, params).
    z_grid : np.ndarray
        Monotonically increasing array of z values.
    y0 : np.ndarray
        Initial state vector at z = z_grid[0].
    params : object
        Arbitrary parameter container passed to f.
    save_every : int, optional
        Save state every N steps (default: 1).
    check_nan : bool, optional
        Check for NaN/Inf during integration (default: True).

    Returns
    -------
    z_out : np.ndarray
        z values at which the solution is stored.
    y_out : np.ndarray
        Stored solution array with shape (N_saved, state_dim).
    """

    z_grid = np.asarray(z_grid, dtype=float)

    if z_grid.ndim != 1:
        raise ValueError("z_grid must be a one-dimensional array")

    if save_every <= 0:
        raise ValueError("save_every must be a positive integer")

    n_steps = len(z_grid) - 1
    state_dim = y0.size

    # Preallocate output (upper bound on size)
    n_saved = n_steps // save_every + 1
    z_out = np.empty(n_saved, dtype=float)
    y_out = np.empty((n_saved, state_dim), dtype=y0.dtype)

    # Initial state
    y = y0.copy()
    z_out[0] = z_grid[0]
    y_out[0] = y

    save_idx = 1

    for i in range(n_steps):
        z = z_grid[i]
        dz = z_grid[i + 1] - z_grid[i]

        y = rk4_step(f, z, y, dz, params)

        if check_nan and not np.all(np.isfinite(y)):
            raise FloatingPointError(
                f"NaN or Inf detected at step {i}, z = {z}"
            )

        if (i + 1) % save_every == 0:
            z_out[save_idx] = z_grid[i + 1]
            y_out[save_idx] = y
            save_idx += 1

    return z_out[:save_idx], y_out[:save_idx]


# ---------------------------------------------------------------------
# Convenience wrapper: integrate on [0, z_max] with fixed dz
# ---------------------------------------------------------------------

def integrate_interval(
    f: RHSFunction,
    z_max: float,
    dz: float,
    y0: np.ndarray,
    params: object,
    *,
    save_every: int = 1,
    check_nan: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate an ODE system on the interval [0, z_max] with fixed step dz.

    Parameters
    ----------
    f : callable
        RHS function f(z, y, params).
    z_max : float
        Upper integration limit.
    dz : float
        Step size.
    y0 : np.ndarray
        Initial state vector.
    params : object
        Arbitrary parameter container passed to f.

    Returns
    -------
    z_out : np.ndarray
        z values at which the solution is stored.
    y_out : np.ndarray
        Stored solution array.
    """

    if z_max <= 0.0:
        raise ValueError("z_max must be positive")

    if dz <= 0.0:
        raise ValueError("dz must be positive")

    n_steps = int(round(z_max / dz))
    z_grid = np.linspace(0.0, z_max, n_steps + 1)

    return integrate_fixed_step(
        f=f,
        z_grid=z_grid,
        y0=y0,
        params=params,
        save_every=save_every,
        check_nan=check_nan,
    )
