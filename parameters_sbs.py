"""
parameters_sbs.py

Conventions:
- z is measured in km.
- gamma is in 1/(W·km), alpha and beta are in 1/km.
- GammaB is in 1/s.
- vA is in km/s (so GammaB/(2*vA) has units 1/km).
- Frequencies (omega, OmegaB, DeltaOmega) are in rad/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

# Import your existing parameter groups to keep everything compatible.
from parameters import FiberParams, WaveParams


# ---------------------------------------------------------------------
# SBS parameters (SIMPLIFIED: kappa1, kappa2 are scalars)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SBSParams:
    """
    Parameters required by the SBS coupling terms (simplified).

    kappa1: scalar coupling used for ALL j in the A_j and B_j equations.
    kappa2: scalar coupling used for ALL j in the Q_j equation.

    OmegaB: Brillouin frequency shift [rad/s]
    GammaB: acoustic damping rate [1/s]
    vA: acoustic velocity [km/s]

    omega_B: backward-wave angular frequencies [rad/s], length 4
    """

    kappa1: float         # scalar
    kappa2: float         # scalar
    OmegaB: float         # [rad/s]
    GammaB: float         # [1/s]
    vA: float             # [km/s]
    omega_B: np.ndarray   # shape (4,), float


# ---------------------------------------------------------------------
# SBS boundary conditions / initial conditions
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SBSBoundaryConditions:
    """
    Boundary conditions for the bidirectional SBS problem.

    A0: forward optical fields at z=0
    B_L: backward optical fields at z=L (end of fiber)
    Q0: acoustic envelopes at z=0 (often zeros)

    All arrays must have shape (4,) and dtype complex.
    """

    A0: np.ndarray     # complex, shape (4,)
    B_L: np.ndarray    # complex, shape (4,)
    Q0: np.ndarray     # complex, shape (4,)


# ---------------------------------------------------------------------
# Combined model parameters (optional but convenient)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FWM_SBS_ModelParams:
    """
    Full parameter bundle for the coupled FWM+SBS solver.
    """

    fiber: FiberParams
    waves: WaveParams
    sbs: SBSParams
    bc: SBSBoundaryConditions


# ---------------------------------------------------------------------
# Helpers (validation, construction)
# ---------------------------------------------------------------------

def _as_len4_float(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be an array of length 4, got shape {arr.shape}")
    return arr


def _as_len4_complex(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be a complex array of length 4, got shape {arr.shape}")
    return arr


def _validate_positive(value: float, name: str, allow_zero: bool = False) -> None:
    if allow_zero:
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    else:
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}")


def mps_to_kmps(v_m_per_s: float) -> float:
    """
    Convert acoustic velocity from m/s to km/s.
    """
    return float(v_m_per_s) * 1e-3


def compute_delta_omega(
    *,
    OmegaB: float,
    omega_A: np.ndarray,
    omega_B: np.ndarray,
) -> np.ndarray:
    """
    DeltaOmega_j = OmegaB - (omega_Aj - omega_Bj)
    Returns array length 4 [rad/s].
    """
    omega_A = _as_len4_float(omega_A, "omega_A")
    omega_B = _as_len4_float(omega_B, "omega_B")
    return float(OmegaB) - (omega_A - omega_B)


def complex_from_power_phase(
    P: np.ndarray,
    phase: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Construct complex amplitudes from powers and phases.

    Uses the same normalization as your current code: |A|^2 = P.
    So amplitude = sqrt(P) * exp(i*phase).

    Parameters
    ----------
    P : array_like, shape (4,)
        Powers in watts (must be >= 0).
    phase : array_like, shape (4,), optional
        Phase in radians. If None, all phases are 0.

    Returns
    -------
    A : np.ndarray, complex128, shape (4,)
    """
    P = _as_len4_float(P, "P")
    if np.any(P < 0.0):
        raise ValueError("P must be non-negative")

    if phase is None:
        phase_arr = np.zeros(4, dtype=float)
    else:
        phase_arr = _as_len4_float(phase, "phase")

    amp = np.sqrt(P, dtype=np.float64)
    return (amp * np.exp(1j * phase_arr)).astype(np.complex128)


# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------

def make_sbs_params(
    *,
    kappa1: float,
    kappa2: float,
    OmegaB: float,
    GammaB: float,
    vA_km_s: float,
    omega_A: np.ndarray,
    omega_B: Optional[np.ndarray] = None,
) -> SBSParams:
    """
    Create SBSParams with validation (simplified: scalar kappa1, kappa2).

    Parameters
    ----------
    kappa1, kappa2 : float
        SBS coupling coefficients (applied identically for j=1..4).
    OmegaB : float
        Brillouin shift [rad/s]
    GammaB : float
        Acoustic damping rate [1/s]
    vA_km_s : float
        Acoustic velocity [km/s] (IMPORTANT: km/s to match z in km)
    omega_A : array_like (4,)
        Forward-wave angular frequencies [rad/s] (typically WaveParams.omega)
    omega_B : array_like (4,), optional
        Backward-wave angular frequencies [rad/s].
        If None, default is omega_B = omega_A - OmegaB.

    Returns
    -------
    SBSParams
    """
    omega_A = _as_len4_float(omega_A, "omega_A")

    _validate_positive(float(GammaB), "GammaB")
    _validate_positive(float(vA_km_s), "vA_km_s")

    # kappa can be 0 for testing "no SBS" regimes
    _validate_positive(float(kappa1), "kappa1", allow_zero=True)
    _validate_positive(float(kappa2), "kappa2", allow_zero=True)

    if omega_B is None:
        omega_B_arr = (omega_A - float(OmegaB)).astype(float)
    else:
        omega_B_arr = _as_len4_float(omega_B, "omega_B")

    return SBSParams(
        kappa1=float(kappa1),
        kappa2=float(kappa2),
        OmegaB=float(OmegaB),
        GammaB=float(GammaB),
        vA=float(vA_km_s),
        omega_B=omega_B_arr,
    )


def make_sbs_params_from_m_per_s(
    *,
    kappa1: float,
    kappa2: float,
    OmegaB: float,
    GammaB: float,
    vA_m_s: float,
    omega_A: np.ndarray,
    omega_B: Optional[np.ndarray] = None,
) -> SBSParams:
    """
    Same as make_sbs_params, but accepts vA in m/s and converts to km/s.
    """
    vA_km_s = mps_to_kmps(float(vA_m_s))
    return make_sbs_params(
        kappa1=kappa1,
        kappa2=kappa2,
        OmegaB=OmegaB,
        GammaB=GammaB,
        vA_km_s=vA_km_s,
        omega_A=omega_A,
        omega_B=omega_B,
    )


def make_sbs_boundary_conditions(
    *,
    A0: np.ndarray,
    B_L: np.ndarray,
    Q0: Optional[np.ndarray] = None,
) -> SBSBoundaryConditions:
    """
    Create SBSBoundaryConditions with validation.

    Q0 defaults to zeros if not provided.
    """
    A0 = _as_len4_complex(A0, "A0")
    B_L = _as_len4_complex(B_L, "B_L")

    if Q0 is None:
        Q0_arr = np.zeros(4, dtype=np.complex128)
    else:
        Q0_arr = _as_len4_complex(Q0, "Q0")

    return SBSBoundaryConditions(A0=A0, B_L=B_L, Q0=Q0_arr)


def make_sbs_boundary_conditions_from_powers(
    *,
    P_A0: np.ndarray,
    phase_A0: Optional[np.ndarray] = None,
    P_B_L: Optional[np.ndarray] = None,
    phase_B_L: Optional[np.ndarray] = None,
    Q0: Optional[np.ndarray] = None,
) -> SBSBoundaryConditions:
    """
    Convenience constructor using power/phase for A0 and B_L.

    If P_B_L is None, it defaults to zeros (no injected Stokes at z=L).
    """
    A0 = complex_from_power_phase(P_A0, phase_A0)

    if P_B_L is None:
        B_L = np.zeros(4, dtype=np.complex128)
    else:
        B_L = complex_from_power_phase(P_B_L, phase_B_L)

    return make_sbs_boundary_conditions(A0=A0, B_L=B_L, Q0=Q0)


def make_fwm_sbs_model_params(
    *,
    fiber: FiberParams,
    waves: WaveParams,
    sbs: SBSParams,
    bc: SBSBoundaryConditions,
) -> FWM_SBS_ModelParams:
    """
    Combine existing fiber/wave params with SBS params and boundary conditions.
    """
    if waves.omega.shape != (4,):
        raise ValueError("waves.omega must have shape (4,)")
    if sbs.omega_B.shape != (4,):
        raise ValueError("sbs.omega_B must have shape (4,)")

    return FWM_SBS_ModelParams(
        fiber=fiber,
        waves=waves,
        sbs=sbs,
        bc=bc,
    )


# ---------------------------------------------------------------------
# Derived quantities (often used by the RHS)
# ---------------------------------------------------------------------

def sbs_decay_per_km(sbs: SBSParams) -> float:
    """
    Return GammaB/(2*vA) in units of 1/km, consistent with z in km.
    """
    return sbs.GammaB / (2.0 * sbs.vA)


def sbs_delta_omega(
    *,
    sbs: SBSParams,
    omega_A: np.ndarray,
) -> np.ndarray:
    """
    Compute DeltaOmega_j for j=1..4:
      DeltaOmega_j = OmegaB - (omega_Aj - omega_Bj)
    """
    return compute_delta_omega(OmegaB=sbs.OmegaB, omega_A=omega_A, omega_B=sbs.omega_B)
