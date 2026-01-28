"""
simulation_sbs.py

High-level simulation runner for the coupled scalar FWM + SBS model.

This module should:
- validate SimulationConfig
- assemble physical/model parameter objects (from parameters_sbs.py)
- call the solver (solver_fwm_sbs.py)
- return raw arrays (z, A, B, Q) + optional diagnostics dict

Key change (vs the earlier "all-numerical" approach):
----------------------------------------------------
We now expose `solver_settings` / `forward_method` so you can select the
"fast Q" exact exponential update (variable-change style) for stiff ΓB,
while retaining the option to run the full RK4 evolution for Q.

Assumptions (consistent with the rest of your project):
- z is in km
- gamma is in 1/(W*km)
- alpha is in 1/km
- beta is in 1/km
- fields A,B have units sqrt(W)
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from config import SimulationConfig, validate_config

# Your SBS parameter module (previously implemented)
from parameters_sbs import (
    make_fiber_params_sbs,
    make_wave_params_sbs,
    make_sbs_params,
    make_boundary_conditions_sbs,
    make_initial_conditions_sbs,
    make_fwm_sbs_params,
)

# Your coupled solver (previously implemented / modified)
from solver_fwm_sbs import (
    SolverSettings,
    solve_fwm_sbs,
)


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _as_1d_float(x: np.ndarray | list[float] | tuple[float, ...], n: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return arr


def _as_1d_complex(x: np.ndarray | list[complex] | tuple[complex, ...], n: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return arr


def _fields_from_power_phase(P: np.ndarray, phase: np.ndarray | None, name: str) -> np.ndarray:
    """
    Build complex field vector from powers and (optional) phases.
    Convention: |A_j|^2 = P_j.
    """
    P = np.asarray(P, dtype=float).reshape(-1)
    if np.any(P < 0.0):
        raise ValueError(f"{name} powers must be non-negative")

    amp = np.sqrt(P, dtype=np.complex128)

    if phase is None:
        return amp

    phase = np.asarray(phase, dtype=float).reshape(-1)
    if phase.shape != P.shape:
        raise ValueError(f"{name} phase must have same shape as powers, got {phase.shape} vs {P.shape}")

    return amp * np.exp(1j * phase)


def _maybe_metadata(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    return obj


# ---------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------

def run_single_simulation_fwm_sbs(
    cfg: SimulationConfig,
    *,
    # ---- Kerr/FWM optics ----
    gamma: float,
    alpha: float,
    beta: np.ndarray,      # (4,) 1/km
    omega: np.ndarray,     # (4,) rad/s (kept for completeness; you may not use it in RHS)
    p_in_forward: np.ndarray,   # (4,) W
    p_in_backward: np.ndarray | None = None,  # (4,) W (typically 0 at z=L)
    phase_in_forward: np.ndarray | None = None,   # (4,) rad
    phase_in_backward: np.ndarray | None = None,  # (4,) rad

    # ---- SBS ----
    kappa1: float = 0.0,       # 1/(sqrt(W)*km)
    kappa2: float = 0.0,
    v_a: float = 1.0,          # acoustic velocity scaling in your z-domain model (must match RHS conventions)
    Gamma_B: float = 1.0,      # 1/km in z-domain form after your nondimensionalization (or mapped consistently)
    Omega_B: float = 0.0,      # rad/s (only used to form detunings)
    delta_Omega: np.ndarray | None = None,  # (4,) = Ω_B - (ω_Aj - ω_Bj)  [rad/s] or mapped equivalent
    q0: np.ndarray | None = None,           # (4,) complex acoustic envelopes at starting point (solver handles location)

    # ---- Solver control ----
    solver_settings: SolverSettings | None = None,
    forward_method: str | None = None,      # convenience override: "full" or "expQ"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Run one coupled FWM+SBS simulation.

    Returns
    -------
    z : (N,) float
    A : (N,4) complex    forward optical waves
    B : (N,4) complex    backward/Stokes waves
    Q : (N,4) complex    acoustic waves
    info : dict          diagnostics (iterations, residuals, etc.)
    """

    validate_config(cfg)

    beta = _as_1d_float(beta, 4, "beta")
    omega = _as_1d_float(omega, 4, "omega")
    p_in_forward = _as_1d_float(p_in_forward, 4, "p_in_forward")

    if p_in_backward is None:
        p_in_backward = np.zeros(4, dtype=float)
    else:
        p_in_backward = _as_1d_float(p_in_backward, 4, "p_in_backward")

    A0 = _fields_from_power_phase(p_in_forward, phase_in_forward, "forward")
    B_L = _fields_from_power_phase(p_in_backward, phase_in_backward, "backward")

    if q0 is None:
        q0 = np.zeros(4, dtype=np.complex128)
    else:
        q0 = _as_1d_complex(q0, 4, "q0")

    if delta_Omega is None:
        delta_Omega = np.zeros(4, dtype=float)
    else:
        # Required only for modelling detuning
        delta_Omega = _as_1d_float(delta_Omega, 4, "delta_Omega")

    # ---- Build parameter objects (duck-typed by the model/solver) ----
    fiber = make_fiber_params_sbs(
        gamma=float(gamma),
        alpha=float(alpha),
        beta=beta,
    )

    waves = make_wave_params_sbs(
        omega=omega,
        P_in_forward=p_in_forward,
        P_in_backward=p_in_backward,
    )

    sbs = make_sbs_params(
        kappa1=float(kappa1),
        kappa2=float(kappa2),
        v_a=float(v_a),
        Gamma_B=float(Gamma_B),
        Omega_B=float(Omega_B),
        delta_Omega=delta_Omega,
    )

    bc = make_boundary_conditions_sbs(
        A0=A0,
        B_L=B_L,
    )

    ic = make_initial_conditions_sbs(
        Q0=q0,
    )

    params = make_fwm_sbs_params(
        fiber=fiber,
        waves=waves,
        sbs=sbs,
        bc=bc,
        ic=ic,
    )

    # ---- Solver settings: default to the fast-Q method unless user explicitly forces otherwise ----
    if solver_settings is None:
        if forward_method is None:
            forward_method = "expQ"  # recommended default for stiff ΓB
        solver_settings = SolverSettings(forward_method=forward_method)
    else:
        # Allow a convenience override while keeping the rest of the settings intact
        if forward_method is not None:
            solver_settings = SolverSettings(**{**asdict(solver_settings), "forward_method": forward_method})

    z, A, B, Q, info = solve_fwm_sbs(cfg=cfg, params=params, settings=solver_settings)

    # Add some lightweight provenance to info
    info = dict(info) if isinstance(info, dict) else {"solver_info": info}
    info["cfg"] = _maybe_metadata(cfg)
    info["solver_settings"] = _maybe_metadata(solver_settings)

    return z, A, B, Q, info


# ---------------------------------------------------------------------
# Minimal example (optional): kept small on purpose
# ---------------------------------------------------------------------

def example_seeded_signal_idler_sbs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    A small default example:
    - moderately strong pumps
    - seeded signal + idler
    - default fast-Q exponential update

    You can call this from a SBS-specific main.py.
    """

    cfg = SimulationConfig(
        z_max=0.2,     # km
        dz=2e-4,       # km (200 m step) -> adjust as you like
        integrator="rk4",
        save_every=10,
        check_nan=True,
        verbose=False,
    )

    gamma = 10.0
    alpha = 0.0

    beta0 = 5.8e9  # 1/km (dummy baseline)
    beta = beta0 * np.ones(4, dtype=float)

    omega0 = 2.0 * np.pi * 193.5e12  # ~1550 nm carrier (rough)
    omega = omega0 * np.ones(4, dtype=float)

    p_in_fwd = np.array([1.0, 1.0, 1e-3, 1e-3], dtype=float)  # W
    p_in_bwd = np.zeros(4, dtype=float)  # no injected Stokes at z=L

    # SBS knobs (placeholders in the "engineering sense": choose your calibrated values)
    kappa1 = 1e-3
    kappa2 = 1e-3
    v_a = 1.0
    Gamma_B = 5.0
    Omega_B = 0.0
    delta_Omega = np.zeros(4, dtype=float)

    return run_single_simulation_fwm_sbs(
        cfg,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        omega=omega,
        p_in_forward=p_in_fwd,
        p_in_backward=p_in_bwd,
        kappa1=kappa1,
        kappa2=kappa2,
        v_a=v_a,
        Gamma_B=Gamma_B,
        Omega_B=Omega_B,
        delta_Omega=delta_Omega,
        solver_settings=SolverSettings(forward_method="expQ"),
    )
