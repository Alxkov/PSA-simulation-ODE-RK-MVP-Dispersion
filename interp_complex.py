"""
interp_complex.py

Fast, minimal utilities for interpolating complex-valued arrays on a 1D grid.

Why this exists:
- numpy.interp works only on real arrays.
- RK4 evaluates RHS at intermediate points, so for the FWM+SBS forward/backward
  sweeps you need B(z) from a stored grid, and later A(z), Q(z) from their grids.

Design goals:
- Robust to both increasing and decreasing z_grids.
- Supports:
  (a) single complex scalar series: y(z_grid) -> y(z0)
  (b) multiple channels: Y(z_grid, n_ch) -> Y(z0, :)
  (c) multiple queries: z_query array -> outputs on those points

Conventions:
- z_grid: 1D float array, monotonic (either increasing or decreasing).
- Values: complex array with first axis matching z_grid length.
"""

from __future__ import annotations

from typing import Union, Optional
import numpy as np


ArrayLike = Union[float, np.ndarray]


def _ensure_1d_float(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {arr.shape}")
    if arr.size < 2:
        raise ValueError(f"{name} must have at least 2 points, got {arr.size}")
    return arr


def _ensure_complex_values(values: np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape[0] != n:
        raise ValueError(
            f"{name} first dimension must match z_grid length {n}, got {arr.shape[0]}"
        )
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128, copy=False)
    return arr


def _is_monotonic(z: np.ndarray) -> bool:
    dz = np.diff(z)
    return np.all(dz > 0) or np.all(dz < 0)


def _to_increasing_grid(
    z_grid: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure z_grid is increasing. If decreasing, flip both z_grid and values.
    """
    if z_grid[0] < z_grid[-1]:
        return z_grid, values
    return z_grid[::-1].copy(), values[::-1].copy()


def interp_complex(
    z0: float,
    z_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    left: Optional[complex] = None,
    right: Optional[complex] = None,
) -> complex:
    """
    Interpolate a complex scalar series y(z_grid) at a single point z0.

    Parameters
    ----------
    z0 : float
        Query coordinate.
    z_grid : np.ndarray, shape (N,)
        Monotonic grid (increasing or decreasing).
    y_grid : np.ndarray, shape (N,)
        Complex samples corresponding to z_grid.
    left, right : complex, optional
        Values returned for extrapolation beyond the grid.
        If None, clamp to boundary value (like "hold").

    Returns
    -------
    complex
        Interpolated value at z0.
    """
    z = _ensure_1d_float(z_grid, "z_grid")
    y = _ensure_complex_values(y_grid, z.size, "y_grid")
    if not _is_monotonic(z):
        raise ValueError("z_grid must be strictly monotonic (increasing or decreasing)")

    z_inc, y_inc = _to_increasing_grid(z, y)

    if left is None:
        left_r = float(np.real(y_inc[0]))
        left_i = float(np.imag(y_inc[0]))
    else:
        left_r = float(np.real(left))
        left_i = float(np.imag(left))

    if right is None:
        right_r = float(np.real(y_inc[-1]))
        right_i = float(np.imag(y_inc[-1]))
    else:
        right_r = float(np.real(right))
        right_i = float(np.imag(right))

    yr = np.interp(float(z0), z_inc, np.real(y_inc), left=left_r, right=right_r)
    yi = np.interp(float(z0), z_inc, np.imag(y_inc), left=left_i, right=right_i)
    return complex(yr, yi)


def interp_complex_vec(
    z0: float,
    z_grid: np.ndarray,
    Y_grid: np.ndarray,
    *,
    left: Optional[np.ndarray] = None,
    right: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolate multi-channel complex data at a single point z0.

    Parameters
    ----------
    z0 : float
        Query coordinate.
    z_grid : np.ndarray, shape (N,)
        Monotonic grid.
    Y_grid : np.ndarray, shape (N, M)
        Complex samples, M channels.
    left, right : np.ndarray, optional, shape (M,)
        Extrapolation values. If None, clamps to boundary row.

    Returns
    -------
    np.ndarray, shape (M,), complex128
        Interpolated vector at z0.
    """
    z = _ensure_1d_float(z_grid, "z_grid")
    Y = _ensure_complex_values(Y_grid, z.size, "Y_grid")
    if Y.ndim != 2:
        raise ValueError(f"Y_grid must be 2D (N,M), got shape {Y.shape}")
    if not _is_monotonic(z):
        raise ValueError("z_grid must be strictly monotonic (increasing or decreasing)")

    z_inc, Y_inc = _to_increasing_grid(z, Y)
    M = Y_inc.shape[1]

    if left is None:
        left_c = Y_inc[0, :]
    else:
        left_arr = np.asarray(left)
        if left_arr.shape != (M,):
            raise ValueError(f"left must have shape ({M},), got {left_arr.shape}")
        left_c = left_arr.astype(np.complex128, copy=False)

    if right is None:
        right_c = Y_inc[-1, :]
    else:
        right_arr = np.asarray(right)
        if right_arr.shape != (M,):
            raise ValueError(f"right must have shape ({M},), got {right_arr.shape}")
        right_c = right_arr.astype(np.complex128, copy=False)

    # Interpolate each channel. M is small (4 or 8), loop is fine and clear.
    out = np.empty((M,), dtype=np.complex128)
    zq = float(z0)
    for j in range(M):
        yr = np.interp(
            zq, z_inc, np.real(Y_inc[:, j]),
            left=float(np.real(left_c[j])),
            right=float(np.real(right_c[j])),
        )
        yi = np.interp(
            zq, z_inc, np.imag(Y_inc[:, j]),
            left=float(np.imag(left_c[j])),
            right=float(np.imag(right_c[j])),
        )
        out[j] = yr + 1j * yi
    return out


def interp_complex_many(
    z_query: np.ndarray,
    z_grid: np.ndarray,
    y_grid: np.ndarray,
    *,
    left: Optional[complex] = None,
    right: Optional[complex] = None,
) -> np.ndarray:
    """
    Interpolate a complex scalar series at many query points.

    Parameters
    ----------
    z_query : np.ndarray, shape (K,)
        Query coordinates.
    z_grid : np.ndarray, shape (N,)
        Monotonic grid.
    y_grid : np.ndarray, shape (N,)
        Complex samples.
    left, right : complex, optional
        Extrapolation values. If None, clamps to boundary.

    Returns
    -------
    np.ndarray, shape (K,), complex128
    """
    zq = np.asarray(z_query, dtype=float)
    if zq.ndim != 1:
        raise ValueError(f"z_query must be 1D, got shape {zq.shape}")

    z = _ensure_1d_float(z_grid, "z_grid")
    y = _ensure_complex_values(y_grid, z.size, "y_grid")
    if not _is_monotonic(z):
        raise ValueError("z_grid must be strictly monotonic (increasing or decreasing)")

    z_inc, y_inc = _to_increasing_grid(z, y)

    if left is None:
        left_r = float(np.real(y_inc[0]))
        left_i = float(np.imag(y_inc[0]))
    else:
        left_r = float(np.real(left))
        left_i = float(np.imag(left))

    if right is None:
        right_r = float(np.real(y_inc[-1]))
        right_i = float(np.imag(y_inc[-1]))
    else:
        right_r = float(np.real(right))
        right_i = float(np.imag(right))

    yr = np.interp(zq, z_inc, np.real(y_inc), left=left_r, right=right_r)
    yi = np.interp(zq, z_inc, np.imag(y_inc), left=left_i, right=right_i)
    return (yr + 1j * yi).astype(np.complex128, copy=False)


def interp_complex_vec_many(
    z_query: np.ndarray,
    z_grid: np.ndarray,
    Y_grid: np.ndarray,
    *,
    left: Optional[np.ndarray] = None,
    right: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Interpolate multi-channel complex data at many query points.

    Parameters
    ----------
    z_query : np.ndarray, shape (K,)
        Query coordinates.
    z_grid : np.ndarray, shape (N,)
        Monotonic grid.
    Y_grid : np.ndarray, shape (N, M)
        Complex samples.
    left, right : np.ndarray, optional, shape (M,)
        Extrapolation values. If None, clamps to boundary rows.

    Returns
    -------
    np.ndarray, shape (K, M), complex128
    """
    zq = np.asarray(z_query, dtype=float)
    if zq.ndim != 1:
        raise ValueError(f"z_query must be 1D, got shape {zq.shape}")

    z = _ensure_1d_float(z_grid, "z_grid")
    Y = _ensure_complex_values(Y_grid, z.size, "Y_grid")
    if Y.ndim != 2:
        raise ValueError(f"Y_grid must be 2D (N,M), got shape {Y.shape}")
    if not _is_monotonic(z):
        raise ValueError("z_grid must be strictly monotonic (increasing or decreasing)")

    z_inc, Y_inc = _to_increasing_grid(z, Y)
    M = Y_inc.shape[1]

    if left is None:
        left_c = Y_inc[0, :]
    else:
        left_arr = np.asarray(left)
        if left_arr.shape != (M,):
            raise ValueError(f"left must have shape ({M},), got {left_arr.shape}")
        left_c = left_arr.astype(np.complex128, copy=False)

    if right is None:
        right_c = Y_inc[-1, :]
    else:
        right_arr = np.asarray(right)
        if right_arr.shape != (M,):
            raise ValueError(f"right must have shape ({M},), got {right_arr.shape}")
        right_c = right_arr.astype(np.complex128, copy=False)

    out = np.empty((zq.size, M), dtype=np.complex128)
    for j in range(M):
        yr = np.interp(
            zq, z_inc, np.real(Y_inc[:, j]),
            left=float(np.real(left_c[j])),
            right=float(np.real(right_c[j])),
        )
        yi = np.interp(
            zq, z_inc, np.imag(Y_inc[:, j]),
            left=float(np.imag(left_c[j])),
            right=float(np.imag(right_c[j])),
        )
        out[:, j] = yr + 1j * yi
    return out
