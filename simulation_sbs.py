"""
simulation_sbs.py

High-level simulation runner for the coupled Scalar FWM + SBS model.

This module defines HOW SBS simulations are run:
- assembling parameters (fiber/waves/SBS/boundaries)
- calling the bidirectional fixed-point solver
- returning raw results

It is intentionally parallel to simulation.py, and does NOT change the pure-FWM path.

"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import numpy as np

import constants
from math import pi

from config import SimulationConfig, default_simulation_config, custom_simulation_config, validate_config
from parameters import make_fiber_params, make_wave_params

from parameters_sbs import (
    SBSParams,
    SBSBoundaryConditions,
    FWM_SBS_ModelParams,
    make_sbs_params,
    make_sbs_params_from_m_per_s,
    make_sbs_boundary_conditions,
    make_sbs_boundary_conditions_from_powers,
    make_fwm_sbs_model_params,
)

from solver_fwm_sbs import solve_fwm_sbs, SolverSettings


# ---------------------------------------------------------------------
# Core SBS simulation runners
# ---------------------------------------------------------------------

def run_single_simulation_sbs_fields(
    cfg: SimulationConfig,
    *,
    gamma: float,
    alpha: float,
    beta: np.ndarray,
    omega: np.ndarray,
    # Boundary fields
    A0: np.ndarray,                # complex, shape (4,)
    B_L: np.ndarray,               # complex, shape (4,)
    Q0: Optional[np.ndarray] = None,  # complex, shape (4,) or None -> zeros
    # SBS parameters (simplified)
    kappa1: float,
    kappa2: float,
    OmegaB: float,                 # rad/s
    GammaB: float,                 # 1/s
    vA_km_s: float,                # km/s
    omega_B: Optional[np.ndarray] = None,  # rad/s, shape (4,) (defaults to omega - OmegaB)
    # Solver settings
    solver_settings: Optional[SolverSettings] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Run a single coupled FWM+SBS simulation using complex boundary fields.

    Parameters
    ----------
    cfg : SimulationConfig
        Numerical config (z_max [km], dz [km], save_every, etc.)
    gamma, alpha, beta : fiber parameters
        gamma [1/(W·km)], alpha [1/km], beta_j [1/km], shape (4,)
    omega : np.ndarray
        Forward angular frequencies omega_j [rad/s], shape (4,)

    A0 : np.ndarray
        Forward fields at z=0, complex, shape (4,)
    B_L : np.ndarray
        Backward fields at z=L, complex, shape (4,)
    Q0 : np.ndarray | None
        Acoustic envelopes at z=0, complex, shape (4,). If None, zeros.

    kappa1, kappa2 : float
        SBS couplings (same for all 4 channels)
    OmegaB : float
        Brillouin shift [rad/s]
    GammaB : float
        Acoustic damping rate [1/s]
    vA_km_s : float
        Acoustic velocity [km/s]
    omega_B : np.ndarray | None
        Backward angular frequencies [rad/s], shape (4,).
        If None: omega_B = omega - OmegaB (so DeltaOmega = 0).

    solver_settings : SolverSettings | None
        Fixed-point iteration configuration (max_iter, tol_rel, relax, etc.)

    Returns
    -------
    z : np.ndarray, shape (N,)
    A : np.ndarray, shape (N,4)
    B : np.ndarray, shape (N,4)
    Q : np.ndarray, shape (N,4)
    info : dict
        convergence information and useful diagnostics
    """
    validate_config(cfg)

    beta = np.asarray(beta, dtype=float)
    omega = np.asarray(omega, dtype=float)

    if beta.shape != (4,):
        raise ValueError(f"beta must have shape (4,), got {beta.shape}")
    if omega.shape != (4,):
        raise ValueError(f"omega must have shape (4,), got {omega.shape}")

    # --- Build fiber and wave params (reuse existing infrastructure) ---
    fiber = make_fiber_params(gamma=float(gamma), alpha=float(alpha), beta=beta)
    waves = make_wave_params(omega=omega, P_in=np.abs(np.asarray(A0)) ** 2)

    # --- Build SBS params (simplified: scalar kappa1, kappa2) ---
    sbs: SBSParams = make_sbs_params(
        kappa1=float(kappa1),
        kappa2=float(kappa2),
        OmegaB=float(OmegaB),
        GammaB=float(GammaB),
        vA_km_s=float(vA_km_s),
        omega_A=omega,
        omega_B=omega_B,
    )

    # --- Boundary conditions ---
    bc: SBSBoundaryConditions = make_sbs_boundary_conditions(
        A0=np.asarray(A0, dtype=np.complex128),
        B_L=np.asarray(B_L, dtype=np.complex128),
        Q0=None if Q0 is None else np.asarray(Q0, dtype=np.complex128),
    )

    # --- Bundle full model params ---
    params: FWM_SBS_ModelParams = make_fwm_sbs_model_params(
        fiber=fiber,
        waves=waves,
        sbs=sbs,
        bc=bc,
    )

    # --- Solve ---
    z, A, B, Q, info = solve_fwm_sbs(
        cfg=cfg,
        params=params,
        settings=solver_settings,
    )

    return z, A, B, Q, info


def run_single_simulation_sbs(
    cfg: SimulationConfig,
    *,
    gamma: float,
    alpha: float,
    beta: np.ndarray,
    omega: np.ndarray,
    # Boundary powers/phases
    P_A0: np.ndarray,                      # W, shape (4,)
    phase_A0: Optional[np.ndarray] = None, # rad, shape (4,)
    P_B_L: Optional[np.ndarray] = None,    # W, shape (4,), default zeros
    phase_B_L: Optional[np.ndarray] = None,# rad, shape (4,)
    Q0: Optional[np.ndarray] = None,       # complex, shape (4,) or None->zeros
    # SBS parameters
    kappa1: float = 0.0,
    kappa2: float = 0.0,
    OmegaB: float = 0.0,                   # rad/s
    GammaB: float = 0.0,                   # 1/s
    vA_km_s: float = 0.0,                  # km/s
    omega_B: Optional[np.ndarray] = None,
    # Solver settings
    solver_settings: Optional[SolverSettings] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Same as run_single_simulation_sbs_fields, but takes powers and phases for A0 and B_L.

    Note:
    - If P_B_L is None, backward injection is set to zero at z=L.
    - If phase_* is None, phases are assumed zero.
    - If omega_B is None, omega_B = omega - OmegaB.
    """
    validate_config(cfg)

    omega = np.asarray(omega, dtype=float)
    if omega.shape != (4,):
        raise ValueError(f"omega must have shape (4,), got {omega.shape}")

    # --- Boundary conditions from powers/phases ---
    bc = make_sbs_boundary_conditions_from_powers(
        P_A0=np.asarray(P_A0, dtype=float),
        phase_A0=None if phase_A0 is None else np.asarray(phase_A0, dtype=float),
        P_B_L=None if P_B_L is None else np.asarray(P_B_L, dtype=float),
        phase_B_L=None if phase_B_L is None else np.asarray(phase_B_L, dtype=float),
        Q0=None if Q0 is None else np.asarray(Q0, dtype=np.complex128),
    )

    return run_single_simulation_sbs_fields(
        cfg,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        omega=omega,
        A0=bc.A0,
        B_L=bc.B_L,
        Q0=bc.Q0,
        kappa1=kappa1,
        kappa2=kappa2,
        OmegaB=OmegaB,
        GammaB=GammaB,
        vA_km_s=vA_km_s,
        omega_B=omega_B,
        solver_settings=solver_settings,
    )


# ---------------------------------------------------------------------
# Example SBS simulations (optional but useful)
# ---------------------------------------------------------------------

def example_sbs_only_backward_stokes_seed() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Example:
    Single strong forward pump (A1), tiny backward Stokes seed (B1 at z=L),
    other waves off. This is a sanity check that SBS coupling behaves and converges.

    IMPORTANT:
    - Values below are illustrative defaults to verify code plumbing, not a calibrated fiber.
    """
    cfg = custom_simulation_config(z_max=0.5, dz=1e-3, save_every=5, verbose=True)

    # Kerr parameters
    gamma = 10.0     # 1/(W·km)
    alpha = 0.0
    beta = 5.8e9 * np.ones(4)  # 1/km (placeholder-ish constant beta for toy example)

    # Optical frequencies (just set all equal for toy example)
    omega0 = 2.0 * pi * (constants.c / 1.55e-6)
    omega = omega0 * np.ones(4)

    # Forward input: pump only
    P_A0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # Backward seed at z=L: tiny Stokes
    P_B_L = np.array([1e-9, 0.0, 0.0, 0.0], dtype=float)

    # SBS parameters (illustrative)
    OmegaB = 2.0 * pi * 10.8e9     # rad/s
    GammaB = 2.0 * pi * 30e6       # 1/s (order of magnitude)
    vA_km_s = 5.96e3 * 1e-3        # 5960 m/s -> 5.96 km/s

    kappa1 = 1e-3
    kappa2 = 1e-3

    solver_settings = SolverSettings(max_iter=50, tol_rel=1e-8, relax=0.5, init_B="zeros")

    return run_single_simulation_sbs(
        cfg,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        omega=omega,
        P_A0=P_A0,
        P_B_L=P_B_L,
        kappa1=kappa1,
        kappa2=kappa2,
        OmegaB=OmegaB,
        GammaB=GammaB,
        vA_km_s=vA_km_s,
        solver_settings=solver_settings,
    )
