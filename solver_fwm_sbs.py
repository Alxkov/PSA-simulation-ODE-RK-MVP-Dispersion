"""
solver_fwm_sbs.py

Bidirectional fixed-point solver for the coupled Scalar FWM + SBS model.

Core idea:
- Forward sweep (0 -> L): integrate y_AQ = [A1..A4, Q1..Q4] using a current guess B(z)
- Backward sweep (L -> 0): integrate B1..B4 using the newly computed A(z), Q(z)
- Update B(z) with under-relaxation and iterate until convergence

Conventions:
- z is in km
- vA is in km/s
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

from config import SimulationConfig
from integrators import integrate_fixed_step

from interp_complex import interp_complex_vec
from parameters_sbs import FWM_SBS_ModelParams
from fwm_sbs_model import (
    rhs_forward_AQ,
    rhs_backward_B,
    compute_dbeta,
    compute_delta_omega,
)


@dataclass(frozen=True)
class SolverSettings:
    """
    Settings for the fixed-point iteration.
    """
    max_iter: int = 50
    tol_rel: float = 1e-8
    relax: float = 0.5              # under-relaxation factor in (0,1]
    init_B: str = "zeros"           # 'zeros' or 'constant_end'
    eps_norm: float = 1e-30         # numerical floor for relative error


def _validate_cfg(cfg: SimulationConfig) -> None:
    if cfg.z_max <= 0.0:
        raise ValueError("cfg.z_max must be positive")
    if cfg.dz <= 0.0:
        raise ValueError("cfg.dz must be positive")
    if cfg.save_every <= 0:
        raise ValueError("cfg.save_every must be a positive integer")


def _as_complex_len4(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128, copy=False)
    return arr


def _make_z_grid(z_max: float, dz: float) -> np.ndarray:
    """
    Match integrate_interval grid logic: round(z_max/dz) then linspace(0,z_max,N+1).
    """
    n_steps = int(round(z_max / dz))
    if n_steps < 1:
        raise ValueError("z_max/dz results in fewer than 1 step; increase z_max or decrease dz")
    return np.linspace(0.0, float(z_max), n_steps + 1, dtype=float)


def _pack_AQ(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    A = _as_complex_len4(A, "A")
    Q = _as_complex_len4(Q, "Q")
    return np.concatenate([A, Q]).astype(np.complex128, copy=False)


def _unpack_AQ(y_AQ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_AQ)
    if y.shape != (8,):
        raise ValueError(f"y_AQ must have shape (8,), got {y.shape}")
    if not np.iscomplexobj(y):
        y = y.astype(np.complex128, copy=False)
    return y[:4], y[4:]


def _relative_error(B_new: np.ndarray, B_old: np.ndarray, eps_norm: float) -> float:
    """
    Relative L2 error over the full (N,4) complex array.
    """
    d = B_new - B_old
    num = float(np.linalg.norm(d.ravel()))
    den = float(np.linalg.norm(B_new.ravel()))
    return num / max(den, eps_norm)


def _init_B_guess(
    z_grid: np.ndarray,
    B_L: np.ndarray,
    mode: str,
) -> np.ndarray:
    """
    Return B_guess with shape (N,4) on increasing z_grid.
    Boundary condition enforced: B_guess[-1,:] = B_L.
    """
    N = z_grid.size
    B_L = _as_complex_len4(B_L, "B_L")

    if mode == "zeros":
        B = np.zeros((N, 4), dtype=np.complex128)
        B[-1, :] = B_L
        return B

    if mode == "constant_end":
        B = np.empty((N, 4), dtype=np.complex128)
        B[:, :] = B_L[None, :]
        return B

    raise ValueError(f"Unknown init_B mode '{mode}'. Use 'zeros' or 'constant_end'.")


def solve_fwm_sbs(
    cfg: SimulationConfig,
    params: FWM_SBS_ModelParams,
    *,
    settings: Optional[SolverSettings] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Solve the coupled Scalar FWM + SBS equations using forward/backward iteration.

    Returns
    -------
    z_saved : np.ndarray, shape (N_saved,)
    A_saved : np.ndarray, shape (N_saved,4)
    B_saved : np.ndarray, shape (N_saved,4)
    Q_saved : np.ndarray, shape (N_saved,4)
    info : dict with convergence info
    """
    _validate_cfg(cfg)
    if settings is None:
        settings = SolverSettings()

    if settings.max_iter <= 0:
        raise ValueError("settings.max_iter must be positive")
    if settings.tol_rel <= 0.0:
        raise ValueError("settings.tol_rel must be positive")
    if not (0.0 < settings.relax <= 1.0):
        raise ValueError("settings.relax must be in (0,1]")
    if settings.eps_norm <= 0.0:
        raise ValueError("settings.eps_norm must be positive")

    # Boundary conditions
    A0 = _as_complex_len4(params.bc.A0, "params.bc.A0")
    B_L = _as_complex_len4(params.bc.B_L, "params.bc.B_L")
    Q0 = _as_complex_len4(params.bc.Q0, "params.bc.Q0")

    # Precompute quantities used repeatedly inside RHS
    dbeta = compute_dbeta(np.asarray(params.fiber.beta, dtype=float))
    dOmega = compute_delta_omega(params)  # array len 4

    # Full grid (needed for interpolation during RK4)
    z_grid = _make_z_grid(cfg.z_max, cfg.dz)
    z_grid_rev = z_grid[::-1].copy()  # decreasing from L -> 0

    # Initial guess for B(z)
    B_guess = _init_B_guess(z_grid, B_L, settings.init_B)

    # Storage for the latest forward solution
    A_grid = np.empty((z_grid.size, 4), dtype=np.complex128)
    Q_grid = np.empty((z_grid.size, 4), dtype=np.complex128)

    converged = False
    err = float("inf")

    # ----------------------------
    # Fixed-point iteration loop
    # ----------------------------
    for it in range(settings.max_iter):
        # ---- Forward sweep: integrate y_AQ = [A,Q] from 0 -> L ----
        y0_AQ = _pack_AQ(A0, Q0)

        def f_forward(z: float, y_AQ: np.ndarray, _p: object) -> np.ndarray:
            # Interpolate B_guess at this z
            Bz = interp_complex_vec(z, z_grid, B_guess)
            return rhs_forward_AQ(
                z=z,
                y_AQ=y_AQ,
                params=params,
                B_at_z=Bz,
                dbeta=dbeta,
                delta_omega=dOmega,
            )

        z_fwd, y_AQ_fwd = integrate_fixed_step(
            f=f_forward,
            z_grid=z_grid,
            y0=y0_AQ,
            params=None,
            save_every=1,          # MUST be full grid for interpolation in backward sweep
            check_nan=cfg.check_nan,
        )

        if z_fwd.shape != z_grid.shape:
            raise RuntimeError("Forward integration did not return full grid; expected save_every=1 behavior")

        # Extract A(z), Q(z) on full grid
        A_grid[:, :] = y_AQ_fwd[:, :4]
        Q_grid[:, :] = y_AQ_fwd[:, 4:]

        # ---- Backward sweep: integrate B from L -> 0 on reversed grid ----
        B0_rev = B_L.copy()

        def f_backward(z: float, B: np.ndarray, _p: object) -> np.ndarray:
            # Interpolate A(z), Q(z) from forward sweep
            Az = interp_complex_vec(z, z_grid, A_grid)
            Qz = interp_complex_vec(z, z_grid, Q_grid)
            return rhs_backward_B(
                z=z,
                B=B,
                params=params,
                A_at_z=Az,
                Q_at_z=Qz,
            )

        z_bwd, B_bwd = integrate_fixed_step(
            f=f_backward,
            z_grid=z_grid_rev,
            y0=B0_rev,
            params=None,
            save_every=1,
            check_nan=cfg.check_nan,
        )

        if z_bwd.shape != z_grid_rev.shape:
            raise RuntimeError("Backward integration did not return full grid; expected save_every=1 behavior")

        # Convert B_bwd (defined on decreasing grid) back to increasing z order
        B_new = B_bwd[::-1].copy()  # now aligned with z_grid increasing

        # ---- Convergence check ----
        err = _relative_error(B_new, B_guess, settings.eps_norm)

        if cfg.verbose:
            print(f"[solver_fwm_sbs] iter {it+1}/{settings.max_iter}  rel_err(B) = {err:.3e}")

        # Check convergence
        if err < settings.tol_rel:
            converged = True
            B_guess = B_new  # accept
            break

        # Under-relaxation update
        r = float(settings.relax)
        B_guess = (1.0 - r) * B_guess + r * B_new

        # Ensure boundary at z=L is exactly satisfied after relaxation
        B_guess[-1, :] = B_L

    # Final B is B_guess (increasing grid)
    B_grid = B_guess

    # Downsample for output according to cfg.save_every (and always include last point)
    N = z_grid.size
    step = int(cfg.save_every)
    idx = np.arange(0, N, step, dtype=int)
    if idx[-1] != N - 1:
        idx = np.append(idx, N - 1)

    z_saved = z_grid[idx]
    A_saved = A_grid[idx, :].copy()
    B_saved = B_grid[idx, :].copy()
    Q_saved = Q_grid[idx, :].copy()

    info: Dict[str, Any] = {
        "converged": converged,
        "n_iter": int(it + 1),
        "rel_err_B": float(err),
        "tol_rel": float(settings.tol_rel),
        "relax": float(settings.relax),
        "init_B": str(settings.init_B),
        "dbeta": float(dbeta),
        "delta_omega": np.asarray(dOmega, dtype=float),
    }

    return z_saved, A_saved, B_saved, Q_saved, info
