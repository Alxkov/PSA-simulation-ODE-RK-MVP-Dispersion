"""
simulations.py

High-level simulation scripts for the Yaman FWM / OPA model.

This module defines HOW simulations are run:
- assembling parameters
- calling the integrator
- returning raw results

No physics, no numerical algorithms, no plotting.
"""

from __future__ import annotations

import numpy as np

import constants

from config import SimulationConfig, default_simulation_config, validate_config
from parameters import (
    make_fiber_params,
    make_wave_params,
    make_initial_conditions,
    make_model_params,
)
from integrators import integrate_interval
from yaman_model import rhs_yaman_simplified


# ---------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------

def run_single_simulation(
    cfg: SimulationConfig,
    *,
    gamma: float,
    alpha: float,
    beta: np.ndarray,
    omega: np.ndarray,
    p_in: np.ndarray,
    phase_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a single Yaman FWM simulation.

    Parameters
    ----------
    cfg : SimulationConfig
        Numerical simulation configuration.
    gamma : float
        Kerr nonlinearity coefficient [1/(W·km)].
    alpha : float
        Power attenuation coefficient [1/km].
    beta : np.ndarray
        Propagation constants beta_j [1/km], shape (4,).
    omega : np.ndarray
        Angular frequencies omega_j [rad/s], shape (4,).
    p_in : np.ndarray
        Input powers [W] for the four waves.
    phase_in : np.ndarray or None
        Initial phases [rad]. If None, all phases are zero.

    Returns
    -------
    z : np.ndarray
        z-coordinates where the solution is stored.
    A : np.ndarray
        Complex field amplitudes, shape (N, 4).
    """

    validate_config(cfg)

    # --- Build physical parameters ---
    fiber = make_fiber_params(
        gamma=gamma,
        alpha=alpha,
        beta=beta,
    )

    waves = make_wave_params(
        omega=omega,
        P_in=p_in,
    )

    ic = make_initial_conditions(
        P_in=p_in
    )

    params = make_model_params(
        fiber=fiber,
        waves=waves,
        ic=ic,
    )

    # --- Run integration ---
    z, A = integrate_interval(
        f=rhs_yaman_simplified,
        z_max=cfg.z_max,
        dz=cfg.dz,
        y0=params.ic.A0,
        params=params,
        save_every=cfg.save_every,
        check_nan=cfg.check_nan,
    )

    return z, A


# ---------------------------------------------------------------------
# Example simulations
# ---------------------------------------------------------------------

def example_zero_signal() -> tuple[np.ndarray, np.ndarray]:
    """
    Example 1:
    Two pumps, zero signal and idler at the input.

    This is a minimal smoke test for the solver pipeline.
    """

    cfg = default_simulation_config()

    gamma = 1.3      # 1/(W·km)
    alpha = 0.0         # lossless fiber 1/(W*km)

    beta = 5.8e9 * np.array([1.0, 1.0, 1.0, 1.0])      # rad/km
    omega =  constants.c / 1.55e-6 * np.array([1.0, 1.0, 1.0, 1.0])     # rad/s

    p_in = np.array([
        0.5,    # pump 1
        0.0,    # signal
        0.0,    # idler
        0.5,    # pump 2
    ]) # units: W

    return run_single_simulation(
        cfg,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        omega=omega,
        p_in=p_in,
    )


def example_seeded_signal() -> tuple[np.ndarray, np.ndarray]:
    """
    Example 2:
    Two pumps with a weak seeded signal.

    This is the standard configuration for stimulated FWM / OPA.
    """

    cfg = default_simulation_config()

    gamma = 1.3 # 1/(W * km)
    alpha = 0.0

    beta = 5.8e9 * np.array([1.0, 1.0, 1.0, 1.0])      # rad/km
    omega =  constants.c / 1.55e-6 * np.array([1.0, 1.0, 1.0, 1.0])     # rad/s

    p_in = np.array([
        0.5,      # pump 1
        1e-3,     # signal
        0.0,      # idler
        0.5,      # pump 2
    ])

    phase_in = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
    ])

    return run_single_simulation(
        cfg,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        omega=omega,
        p_in=p_in,
        phase_in=phase_in,
    )
