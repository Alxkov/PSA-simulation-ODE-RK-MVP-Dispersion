"""
parameters.py

Physical parameters for the Yaman FWM / OPA model (equations 2.63–2.64).

This module contains ONLY physical and model parameters.
No numerical methods and no simulation control logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------
# Fiber and nonlinear parameters
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FiberParams:
    """
    Parameters describing the optical fiber.
    """

    gamma: float          # [1/(W·m)] Kerr nonlinearity coefficient
    alpha: float          # [1/m] power attenuation coefficient
    beta: np.ndarray      # [1/m] propagation constants beta(omega_j), j = 1..4


# ---------------------------------------------------------------------
# Optical wave parameters
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class WaveParams:
    """
    Parameters of the interacting optical waves.
    """

    omega: np.ndarray     # [rad/s] angular frequencies [omega1, omega2, omega3, omega4]
    P_in: np.ndarray      # [W] input powers of the four waves


# ---------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class InitialConditions:
    """
    Initial complex amplitudes at z = 0.
    """

    A0: np.ndarray        # complex amplitudes [A1, A2, A3, A4] at z = 0


# ---------------------------------------------------------------------
# Combined model parameters
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ModelParams:
    """
    Full set of parameters required by the Yaman RHS.
    """

    fiber: FiberParams
    waves: WaveParams
    ic: InitialConditions


# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------

def make_fiber_params(
    gamma: float,
    alpha: float,
    beta: np.ndarray,
) -> FiberParams:
    """
    Create FiberParams with basic validation.

    @gamma: 1/(W * km)
    @alpha: 1/km
    @beta: 1/km
    """

    beta = np.asarray(beta, dtype=float)

    if beta.shape != (4,):
        raise ValueError("beta must be an array of length 4")

    return FiberParams(
        gamma=gamma,
        alpha=alpha,
        beta=beta,
    )


def make_wave_params(
    omega: np.ndarray,
    P_in: np.ndarray,
) -> WaveParams:
    """
    Create WaveParams with basic validation.

    @omega: rad/s
    P_in: W
    """

    omega = np.asarray(omega, dtype=float)
    P_in = np.asarray(P_in, dtype=float)

    if omega.shape != (4,):
        raise ValueError("omega must be an array of length 4")

    if P_in.shape != (4,):
        raise ValueError("P_in must be an array of length 4")

    if np.any(P_in < 0.0):
        raise ValueError("Input powers must be non-negative")

    return WaveParams(
        omega=omega,
        P_in=P_in,
    )


def make_initial_conditions(
    P_in: np.ndarray
) -> InitialConditions:
    """
    Construct initial complex amplitudes from input powers and phases.

    Parameters
    ----------
    P_in : array_like
        Input powers [W] for the four waves.

    Returns
    -------
    InitialConditions
    """

    P_in = np.asarray(P_in, dtype=float)


    # Field amplitudes: |A_j|^2 = P_j
    A0 = np.sqrt(P_in)

    return InitialConditions(A0=A0)


def make_model_params(
    fiber: FiberParams,
    waves: WaveParams,
    ic: InitialConditions,
) -> ModelParams:

    """
    Combine all parameter groups into a single container.
    """
    return ModelParams(
        fiber=fiber,
        waves=waves,
        ic=ic,
    )
