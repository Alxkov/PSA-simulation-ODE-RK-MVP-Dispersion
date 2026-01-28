"""
solver_fwm_sbs.py

Bidirectional fixed-point solver for the coupled Scalar FWM + SBS model.

This file supports TWO forward-sweep numerical methods:

1) "full"   (original): integrate y_AQ = [A,Q] with RK4 (explicit) on the full grid.
   - Accurate, but stiff when GammaB is large (forces very small dz).

2) "expQ"   (new): integrate A with RK4, but update Q with an exponential (exact-linear)
   step per spatial increment dz (integrating factor method).
   - Removes stiffness from the Q decay term, enabling much larger dz.

Backward sweep for B is unchanged (RK4 on reversed grid), and the overall forward/backward
fixed-point iteration remains unchanged.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Literal

import numpy as np

from config import SimulationConfig
from integrators import integrate_fixed_step, rk4_step

from interp_complex import interp_complex_vec
from parameters_sbs import FWM_SBS_ModelParams
from fwm_sbs_model import (
    rhs_forward_AQ,
    rhs_backward_B,
    compute_dbeta,
    compute_delta_omega,
)


ForwardMethod = Literal["full", "expQ"]


@dataclass(frozen=True)
class SolverSettings:
    """
    Settings for the fixed-point iteration and numerical method selection.

    forward_method:
      - "full": original explicit RK4 integration of [A,Q]
      - "expQ": RK4 for A + exponential (exact-linear) step for Q

    q_forcing:
      Controls how the forcing term F = i*kappa2 * A * conj(B) / vA is evaluated
      inside the exponential update. This is NOT a stiffness requirement; it's accuracy.
      - "left":   use (A_n, B_n)
      - "avg":    use ( (A_n+A_{n+1})/2, (B_n+B_{n+1})/2 )   [default, good]
      - "right":  use (A_{n+1}, B_{n+1})
    """
    max_iter: int = 50
    tol_rel: float = 1e-8
    relax: float = 0.5
    init_B: str = "zeros"                 # 'zeros' or 'constant_end'
    eps_norm: float = 1e-30

    forward_method: ForwardMethod = "full"
    q_forcing: Literal["left", "avg", "right"] = "avg"
    lambda_eps: float = 1e-14             # threshold for |lambda| in expQ update


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
    d = B_new - B_old
    num = float(np.linalg.norm(d.ravel()))
    den = float(np.linalg.norm(B_new.ravel()))
    return num / max(den, eps_norm)


def _init_B_guess(z_grid: np.ndarray, B_L: np.ndarray, mode: str) -> np.ndarray:
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


# -----------------------------------------------------------------------------
# NEW: Exponential update for Q (exact for the linear decay/phase term)
# -----------------------------------------------------------------------------

def _q_lambda(params: FWM_SBS_ModelParams, delta_omega: np.ndarray) -> np.ndarray:
    """
    lambda_j = (GammaB/2 + i*DeltaOmega_j) / vA     [1/km]
    where vA is in km/s and GammaB is in 1/s and DeltaOmega in rad/s.
    """
    GammaB = float(params.sbs.GammaB)
    vA = float(params.sbs.vA)
    if vA <= 0.0:
        raise ValueError(f"params.sbs.vA must be > 0 (km/s), got {vA}")
    dOm = np.asarray(delta_omega, dtype=float)
    if dOm.shape != (4,):
        raise ValueError(f"delta_omega must have shape (4,), got {dOm.shape}")

    return (GammaB / 2.0 + 1j * dOm) / vA


def _q_forcing(params: FWM_SBS_ModelParams, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    F_j(z) = (i*kappa2/vA) * A_j * conj(B_j)    [1/km] * (field units)
    """
    kappa2 = float(params.sbs.kappa2)
    vA = float(params.sbs.vA)
    A = _as_complex_len4(A, "A")
    B = _as_complex_len4(B, "B")
    return (1j * kappa2 * (A * np.conj(B))) / vA


def q_exponential_step(
    Qn: np.ndarray,
    A_used: np.ndarray,
    B_used: np.ndarray,
    dz: float,
    params: FWM_SBS_ModelParams,
    delta_omega: np.ndarray,
    *,
    lambda_eps: float = 1e-14,
) -> np.ndarray:
    """
    Exponential (exact-linear) step for Q over dz:

      dQ/dz = -lambda Q + F   (lambda constant over step if DeltaOmega is constant)
      Q_{n+1} = exp(-lambda dz) Q_n + (1 - exp(-lambda dz))/lambda * F

    Uses expm1 for numerical stability.

    If |lambda| is extremely small (unlikely for SBS), we fall back to:
      Q_{n+1} ≈ Q_n + dz * F
    """
    Qn = _as_complex_len4(Qn, "Qn")
    A_used = _as_complex_len4(A_used, "A_used")
    B_used = _as_complex_len4(B_used, "B_used")

    dz = float(dz)
    if dz == 0.0:
        return Qn.copy()

    lam = _q_lambda(params, delta_omega)          # shape (4,)
    F = _q_forcing(params, A_used, B_used)        # shape (4,)

    # E = exp(-lam*dz)
    E = np.exp(-lam * dz)

    # G = (1 - exp(-lam*dz))/lam = -expm1(-lam*dz)/lam
    # handle tiny |lam|
    G = np.empty((4,), dtype=np.complex128)
    for j in range(4):
        if abs(lam[j]) < float(lambda_eps):
            G[j] = dz  # series: (1 - e^{-x})/x ≈ 1 => G ≈ dz
        else:
            G[j] = -np.expm1(-lam[j] * dz) / lam[j]

    return E * Qn + G * F


# -----------------------------------------------------------------------------
# Forward sweep methods
# -----------------------------------------------------------------------------

def _forward_sweep_full(
    z_grid: np.ndarray,
    A0: np.ndarray,
    Q0: np.ndarray,
    B_guess: np.ndarray,
    params: FWM_SBS_ModelParams,
    dbeta: float,
    delta_omega: np.ndarray,
    *,
    check_nan: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original method: integrate y_AQ=[A,Q] with RK4 on the full grid.
    Returns (A_grid, Q_grid) each shaped (N,4), aligned with increasing z_grid.
    """
    y0_AQ = _pack_AQ(A0, Q0)

    def f_forward(z: float, y_AQ: np.ndarray, _p: object) -> np.ndarray:
        Bz = interp_complex_vec(z, z_grid, B_guess)
        return rhs_forward_AQ(
            z=z,
            y_AQ=y_AQ,
            params=params,
            B_at_z=Bz,
            dbeta=dbeta,
            delta_omega=delta_omega,
        )

    _, y_AQ_fwd = integrate_fixed_step(
        f=f_forward,
        z_grid=z_grid,
        y0=y0_AQ,
        params=None,
        save_every=1,
        check_nan=check_nan,
    )

    A_grid = y_AQ_fwd[:, :4].astype(np.complex128, copy=False)
    Q_grid = y_AQ_fwd[:, 4:].astype(np.complex128, copy=False)
    return A_grid.copy(), Q_grid.copy()


def _rhs_forward_A_only(
    z: float,
    A: np.ndarray,
    Q_fixed: np.ndarray,
    z_grid: np.ndarray,
    B_guess: np.ndarray,
    params: FWM_SBS_ModelParams,
    dbeta: float,
    delta_omega: np.ndarray,
) -> np.ndarray:
    """
    Compute dA/dz while holding Q fixed (provided) and using B_guess(z) interpolation.
    Implemented by calling rhs_forward_AQ and discarding dQ/dz.

    This keeps solver_fwm_sbs.py self-contained (no need to modify fwm_sbs_model.py),
    at the cost of computing dQ/dz internally (minor overhead).
    """
    A = _as_complex_len4(A, "A")
    Q_fixed = _as_complex_len4(Q_fixed, "Q_fixed")

    Bz = interp_complex_vec(z, z_grid, B_guess)
    y_AQ = _pack_AQ(A, Q_fixed)

    dy = rhs_forward_AQ(
        z=z,
        y_AQ=y_AQ,
        params=params,
        B_at_z=Bz,
        dbeta=dbeta,
        delta_omega=delta_omega,
    )
    dA = dy[:4]
    return dA


def _forward_sweep_expQ(
    z_grid: np.ndarray,
    A0: np.ndarray,
    Q0: np.ndarray,
    B_guess: np.ndarray,
    params: FWM_SBS_ModelParams,
    dbeta: float,
    delta_omega: np.ndarray,
    *,
    q_forcing: Literal["left", "avg", "right"],
    lambda_eps: float,
    check_nan: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    New method: RK4 for A, exponential update for Q each step.

    We treat Q as constant during the RK4 step for A (Q_n). This removes stiffness
    while keeping the optical integration high-order. The coupling accuracy is typically
    dominated by the forcing evaluation used in q_exponential_step.

    Returns (A_grid, Q_grid), each shaped (N,4) aligned with increasing z_grid.
    """
    N = z_grid.size
    A_grid = np.empty((N, 4), dtype=np.complex128)
    Q_grid = np.empty((N, 4), dtype=np.complex128)

    A_grid[0, :] = _as_complex_len4(A0, "A0")
    Q_grid[0, :] = _as_complex_len4(Q0, "Q0")

    for n in range(N - 1):
        z_n = float(z_grid[n])
        z_np1 = float(z_grid[n + 1])
        dz = z_np1 - z_n

        A_n = A_grid[n, :]
        Q_n = Q_grid[n, :]

        # Advance A using RK4, holding Q fixed at Q_n
        def fA(z_local: float, A_local: np.ndarray, _p: object) -> np.ndarray:
            return _rhs_forward_A_only(
                z=z_local,
                A=A_local,
                Q_fixed=Q_n,
                z_grid=z_grid,
                B_guess=B_guess,
                params=params,
                dbeta=dbeta,
                delta_omega=delta_omega,
            )

        A_np1 = rk4_step(fA, z_n, A_n, dz, params=None)

        if check_nan:
            if not np.all(np.isfinite(np.real(A_np1))) or not np.all(np.isfinite(np.imag(A_np1))):
                raise ValueError(f"NaN/Inf detected in A at z={z_np1} (expQ forward sweep)")

        A_grid[n + 1, :] = A_np1

        # Prepare B values for forcing evaluation
        B_n = interp_complex_vec(z_n, z_grid, B_guess)
        B_np1 = interp_complex_vec(z_np1, z_grid, B_guess)

        if q_forcing == "left":
            A_used, B_used = A_n, B_n
        elif q_forcing == "right":
            A_used, B_used = A_np1, B_np1
        elif q_forcing == "avg":
            A_used = 0.5 * (A_n + A_np1)
            B_used = 0.5 * (B_n + B_np1)
        else:
            raise ValueError(f"Unknown q_forcing='{q_forcing}'")

        Q_np1 = q_exponential_step(
            Qn=Q_n,
            A_used=A_used,
            B_used=B_used,
            dz=dz,
            params=params,
            delta_omega=delta_omega,
            lambda_eps=lambda_eps,
        )

        if check_nan:
            if not np.all(np.isfinite(np.real(Q_np1))) or not np.all(np.isfinite(np.imag(Q_np1))):
                raise ValueError(f"NaN/Inf detected in Q at z={z_np1} (expQ forward sweep)")

        Q_grid[n + 1, :] = Q_np1

    return A_grid, Q_grid


# -----------------------------------------------------------------------------
# Backward sweep (unchanged)
# -----------------------------------------------------------------------------

def _backward_sweep_B(
    z_grid: np.ndarray,
    B_L: np.ndarray,
    A_grid: np.ndarray,
    Q_grid: np.ndarray,
    params: FWM_SBS_ModelParams,
    *,
    check_nan: bool,
) -> np.ndarray:
    """
    Integrate B from L -> 0 on reversed grid using RK4.
    Returns B_new aligned with increasing z_grid, shape (N,4).
    """
    z_grid_rev = z_grid[::-1].copy()

    def f_backward(z: float, B: np.ndarray, _p: object) -> np.ndarray:
        Az = interp_complex_vec(z, z_grid, A_grid)
        Qz = interp_complex_vec(z, z_grid, Q_grid)
        return rhs_backward_B(
            z=z,
            B=B,
            params=params,
            A_at_z=Az,
            Q_at_z=Qz,
        )

    _, B_bwd = integrate_fixed_step(
        f=f_backward,
        z_grid=z_grid_rev,
        y0=_as_complex_len4(B_L, "B_L"),
        params=None,
        save_every=1,
        check_nan=check_nan,
    )

    # Align with increasing z
    B_new = B_bwd[::-1].copy()
    return B_new


# -----------------------------------------------------------------------------
# Public solver API
# -----------------------------------------------------------------------------

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
    info : dict
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
    if settings.forward_method not in ("full", "expQ"):
        raise ValueError("settings.forward_method must be 'full' or 'expQ'")
    if settings.q_forcing not in ("left", "avg", "right"):
        raise ValueError("settings.q_forcing must be 'left', 'avg', or 'right'")

    # Boundary conditions
    A0 = _as_complex_len4(params.bc.A0, "params.bc.A0")
    B_L = _as_complex_len4(params.bc.B_L, "params.bc.B_L")
    Q0 = _as_complex_len4(params.bc.Q0, "params.bc.Q0")

    # Precompute
    dbeta = compute_dbeta(np.asarray(params.fiber.beta, dtype=float))
    dOmega = compute_delta_omega(params)  # shape (4,)

    # Grids
    z_grid = _make_z_grid(cfg.z_max, cfg.dz)

    # Initial guess for B(z)
    B_guess = _init_B_guess(z_grid, B_L, settings.init_B)

    converged = False
    err = float("inf")

    # latest solutions (full grid)
    A_grid = np.empty((z_grid.size, 4), dtype=np.complex128)
    Q_grid = np.empty((z_grid.size, 4), dtype=np.complex128)

    for it in range(settings.max_iter):
        # ---- Forward sweep ----
        if settings.forward_method == "full":
            A_grid, Q_grid = _forward_sweep_full(
                z_grid=z_grid,
                A0=A0,
                Q0=Q0,
                B_guess=B_guess,
                params=params,
                dbeta=dbeta,
                delta_omega=dOmega,
                check_nan=cfg.check_nan,
            )
        else:  # "expQ"
            A_grid, Q_grid = _forward_sweep_expQ(
                z_grid=z_grid,
                A0=A0,
                Q0=Q0,
                B_guess=B_guess,
                params=params,
                dbeta=dbeta,
                delta_omega=dOmega,
                q_forcing=settings.q_forcing,
                lambda_eps=settings.lambda_eps,
                check_nan=cfg.check_nan,
            )

        # ---- Backward sweep ----
        B_new = _backward_sweep_B(
            z_grid=z_grid,
            B_L=B_L,
            A_grid=A_grid,
            Q_grid=Q_grid,
            params=params,
            check_nan=cfg.check_nan,
        )

        # ---- Convergence check ----
        err = _relative_error(B_new, B_guess, settings.eps_norm)

        if cfg.verbose:
            print(
                f"[solver_fwm_sbs] iter {it+1}/{settings.max_iter}  "
                f"method={settings.forward_method}  rel_err(B)={err:.3e}"
            )

        if err < settings.tol_rel:
            converged = True
            B_guess = B_new
            break

        # Under-relaxation update
        r = float(settings.relax)
        B_guess = (1.0 - r) * B_guess + r * B_new

        # enforce boundary exactly
        B_guess[-1, :] = B_L

    # Final full-grid fields
    B_grid = B_guess

    # Downsample for output
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
        "forward_method": str(settings.forward_method),
        "q_forcing": str(settings.q_forcing),
        "dbeta": float(dbeta),
        "delta_omega": np.asarray(dOmega, dtype=float),
    }

    return z_saved, A_saved, B_saved, Q_saved, info
