"""
parameters_sbs.py

Parameter containers + factory functions for the coupled Scalar FWM + SBS model.

This file is written to be COMPATIBLE with:
1) The original parameter API I proposed earlier:
   - SBSParams, SBSBoundaryConditions, FWM_SBS_ModelParams
   - make_sbs_params(..., vA_km_s=..., GammaB=..., OmegaB=..., omega_A=..., omega_B=...)
   - make_sbs_boundary_conditions(...), make_sbs_boundary_conditions_from_powers(...)

2) The newer simulation_sbs.py variant that imports:
   - make_fiber_params_sbs
   - make_wave_params_sbs
   - make_sbs_params               (but using names v_a, Gamma_B, Omega_B, delta_Omega)
   - make_boundary_conditions_sbs
   - make_initial_conditions_sbs
   - make_fwm_sbs_params

Key physical conventions (recommended, consistent with the solver/model):
- z in km
- gamma in 1/(W·km)
- alpha in 1/km
- beta in 1/km
- vA in km/s
- GammaB in 1/s
- OmegaB and DeltaOmega in rad/s
- Optical fields A,B: |A|^2 and |B|^2 are powers in W
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from parameters import FiberParams, make_fiber_params


# -----------------------------------------------------------------------------
# SBS-specific wave/boundary containers (for the "new" simulation_sbs API)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class WaveParamsSBS:
    """
    Optical wave parameters for SBS model.
    Only omega is required by the RHS; P_in_* are optional provenance.
    """
    omega: np.ndarray              # [rad/s], shape (4,)
    P_in_forward: np.ndarray       # [W], shape (4,)
    P_in_backward: np.ndarray      # [W], shape (4,)


@dataclass(frozen=True)
class BoundaryConditionsSBS:
    """
    Optical boundary conditions:
    - A0 at z=0 (forward waves)
    - B_L at z=L (backward waves)
    """
    A0: np.ndarray                 # complex, shape (4,)
    B_L: np.ndarray                # complex, shape (4,)


@dataclass(frozen=True)
class InitialConditionsSBS:
    """
    Acoustic initial conditions (in this model, Q is integrated forward from z=0).
    """
    Q0: np.ndarray                 # complex, shape (4,)


# -----------------------------------------------------------------------------
# Core SBS params used by the RHS/solver (kappa1/kappa2 are scalars)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class SBSParams:
    """
    Simplified SBS parameters:
    - kappa1 and kappa2 are scalars (same for all 4 channels)
    - omega_B is stored to define detuning via:
        DeltaOmega_j = OmegaB - (omega_Aj - omega_Bj)
    """
    kappa1: float                  # scalar
    kappa2: float                  # scalar
    OmegaB: float                  # [rad/s]
    GammaB: float                  # [1/s]
    vA: float                      # [km/s]
    omega_B: np.ndarray            # [rad/s], shape (4,)


@dataclass(frozen=True)
class SBSBoundaryConditions:
    """
    Boundary conditions expected by solver_fwm_sbs.py:
    - A0 at z=0
    - B_L at z=L
    - Q0 at z=0
    """
    A0: np.ndarray                 # complex, shape (4,)
    B_L: np.ndarray                # complex, shape (4,)
    Q0: np.ndarray                 # complex, shape (4,)


@dataclass(frozen=True)
class FWM_SBS_ModelParams:
    """
    Bundle consumed by fwm_sbs_model.py and solver_fwm_sbs.py.
    """
    fiber: FiberParams
    waves: Union[WaveParamsSBS, object]     # must have .omega (shape (4,))
    sbs: SBSParams
    bc: SBSBoundaryConditions


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _as_len4_float(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be length-4, got shape {arr.shape}")
    return arr


def _as_len4_complex(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.complex128).reshape(-1)
    if arr.shape != (4,):
        raise ValueError(f"{name} must be length-4 complex, got shape {arr.shape}")
    return arr


def _validate_positive(value: float, name: str, allow_zero: bool = False) -> None:
    v = float(value)
    if allow_zero:
        if v < 0.0:
            raise ValueError(f"{name} must be >= 0, got {v}")
    else:
        if v <= 0.0:
            raise ValueError(f"{name} must be > 0, got {v}")


def mps_to_kmps(v_m_s: float) -> float:
    return float(v_m_s) * 1e-3


# -----------------------------------------------------------------------------
# Common helper to create complex fields from powers/phases
# -----------------------------------------------------------------------------

def complex_from_power_phase(P: np.ndarray, phase: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build complex amplitude vector with |A_j|^2 = P_j.

    P: shape (4,), W
    phase: shape (4,), rad (optional)
    """
    P = _as_len4_float(P, "P")
    if np.any(P < 0.0):
        raise ValueError("P must be non-negative")

    amp = np.sqrt(P, dtype=np.complex128)
    if phase is None:
        return amp

    ph = _as_len4_float(phase, "phase")
    return amp * np.exp(1j * ph)


# -----------------------------------------------------------------------------
# Factory functions for the "new" simulation_sbs.py API
# -----------------------------------------------------------------------------

def make_fiber_params_sbs(*, gamma: float, alpha: float, beta: np.ndarray) -> FiberParams:
    """
    Wrapper used by simulation_sbs.py (new API).
    """
    return make_fiber_params(gamma=float(gamma), alpha=float(alpha), beta=np.asarray(beta, dtype=float))


def make_wave_params_sbs(
    *,
    omega: np.ndarray,
    P_in_forward: np.ndarray,
    P_in_backward: Optional[np.ndarray] = None,
) -> WaveParamsSBS:
    omega = _as_len4_float(omega, "omega")
    Pf = _as_len4_float(P_in_forward, "P_in_forward")

    if P_in_backward is None:
        Pb = np.zeros(4, dtype=float)
    else:
        Pb = _as_len4_float(P_in_backward, "P_in_backward")

    if np.any(Pf < 0.0) or np.any(Pb < 0.0):
        raise ValueError("Input powers must be non-negative")

    return WaveParamsSBS(omega=omega, P_in_forward=Pf, P_in_backward=Pb)


def make_boundary_conditions_sbs(*, A0: np.ndarray, B_L: np.ndarray) -> BoundaryConditionsSBS:
    return BoundaryConditionsSBS(A0=_as_len4_complex(A0, "A0"), B_L=_as_len4_complex(B_L, "B_L"))


def make_initial_conditions_sbs(*, Q0: Optional[np.ndarray] = None) -> InitialConditionsSBS:
    if Q0 is None:
        Q0c = np.zeros(4, dtype=np.complex128)
    else:
        Q0c = _as_len4_complex(Q0, "Q0")
    return InitialConditionsSBS(Q0=Q0c)


def make_fwm_sbs_params(
    *,
    fiber: FiberParams,
    waves: WaveParamsSBS,
    sbs: SBSParams,
    bc: BoundaryConditionsSBS,
    ic: InitialConditionsSBS,
) -> FWM_SBS_ModelParams:
    """
    Combine the "new API" boundary+initial containers into the solver's bc container.
    """
    bc_full = SBSBoundaryConditions(A0=bc.A0, B_L=bc.B_L, Q0=ic.Q0)
    return FWM_SBS_ModelParams(fiber=fiber, waves=waves, sbs=sbs, bc=bc_full)


# -----------------------------------------------------------------------------
# SBS params factory (supports BOTH old and new naming styles)
# -----------------------------------------------------------------------------

def make_sbs_params(
    *,
    # Couplings
    kappa1: float,
    kappa2: float,
    # Accept BOTH naming styles:
    OmegaB: Optional[float] = None,
    Omega_B: Optional[float] = None,
    GammaB: Optional[float] = None,
    Gamma_B: Optional[float] = None,
    vA_km_s: Optional[float] = None,
    v_a: Optional[float] = None,
    vA_m_s: Optional[float] = None,
    # Frequencies/detunings
    omega_A: Optional[np.ndarray] = None,          # (4,) rad/s
    omega_B: Optional[np.ndarray] = None,          # (4,) rad/s
    delta_Omega: Optional[np.ndarray] = None,      # (4,) rad/s, DeltaOmega_j = OmegaB - (omega_Aj - omega_Bj)
) -> SBSParams:
    """
    Create SBSParams with validation.


    OLD style (explicit omega_A, vA_km_s):
      make_sbs_params(kappa1=..., kappa2=..., OmegaB=..., GammaB=..., vA_km_s=..., omega_A=..., omega_B=None)

    NEW style (names used in simulation_sbs.py draft):
      make_sbs_params(kappa1=..., kappa2=..., Omega_B=..., Gamma_B=..., v_a=..., delta_Omega=..., omega_A=...)

    omega_B construction rules:
    - If omega_B is provided: use it.
    - Else if omega_A is provided:
        * if delta_Omega is provided:
              omega_Bj = omega_Aj - (OmegaB - delta_Omega_j)
        * else:
              omega_Bj = omega_Aj - OmegaB
    - Else: omega_B defaults to zeros (not recommended, but keeps code from crashing).
    """
    _validate_positive(float(kappa1), "kappa1", allow_zero=True)
    _validate_positive(float(kappa2), "kappa2", allow_zero=True)

    # Resolve parameter names
    if OmegaB is None:
        OmegaB = Omega_B
    if GammaB is None:
        GammaB = Gamma_B
    if vA_km_s is None:
        vA_km_s = v_a

    if vA_km_s is None and vA_m_s is not None:
        vA_km_s = mps_to_kmps(float(vA_m_s))

    if OmegaB is None:
        OmegaB = 0.0
    if GammaB is None:
        GammaB = 0.0
    if vA_km_s is None:
        vA_km_s = 0.0

    # Validate physical positivity (allow zero for "turn off SBS" tests)
    _validate_positive(float(GammaB), "GammaB", allow_zero=True)
    _validate_positive(float(vA_km_s), "vA_km_s", allow_zero=True)

    OmegaB_f = float(OmegaB)
    GammaB_f = float(GammaB)
    vA_f = float(vA_km_s)

    # Prepare omega_B
    if omega_B is not None:
        omega_B_arr = _as_len4_float(omega_B, "omega_B")
    else:
        if omega_A is None:
            omega_B_arr = np.zeros(4, dtype=float)
        else:
            omega_A_arr = _as_len4_float(omega_A, "omega_A")
            if delta_Omega is not None:
                dOm = _as_len4_float(delta_Omega, "delta_Omega")
                # delta_Omega_j = OmegaB - (omega_Aj - omega_Bj)
                # => omega_Bj = omega_Aj - (OmegaB - delta_Omega_j)
                omega_B_arr = omega_A_arr - (OmegaB_f - dOm)
            else:
                omega_B_arr = omega_A_arr - OmegaB_f

    return SBSParams(
        kappa1=float(kappa1),
        kappa2=float(kappa2),
        OmegaB=OmegaB_f,
        GammaB=GammaB_f,
        vA=vA_f,
        omega_B=omega_B_arr,
    )


# -----------------------------------------------------------------------------
# Boundary condition factories (old/original naming retained)
# -----------------------------------------------------------------------------

def make_sbs_boundary_conditions(
    *,
    A0: np.ndarray,
    B_L: np.ndarray,
    Q0: Optional[np.ndarray] = None,
) -> SBSBoundaryConditions:
    A0c = _as_len4_complex(A0, "A0")
    BLc = _as_len4_complex(B_L, "B_L")
    if Q0 is None:
        Q0c = np.zeros(4, dtype=np.complex128)
    else:
        Q0c = _as_len4_complex(Q0, "Q0")
    return SBSBoundaryConditions(A0=A0c, B_L=BLc, Q0=Q0c)


def make_sbs_boundary_conditions_from_powers(
    *,
    P_A0: np.ndarray,
    phase_A0: Optional[np.ndarray] = None,
    P_B_L: Optional[np.ndarray] = None,
    phase_B_L: Optional[np.ndarray] = None,
    Q0: Optional[np.ndarray] = None,
) -> SBSBoundaryConditions:
    A0 = complex_from_power_phase(P_A0, phase_A0)
    if P_B_L is None:
        B_L = np.zeros(4, dtype=np.complex128)
    else:
        B_L = complex_from_power_phase(P_B_L, phase_B_L)
    return make_sbs_boundary_conditions(A0=A0, B_L=B_L, Q0=Q0)


def make_fwm_sbs_model_params(
    *,
    fiber: FiberParams,
    waves: Union[WaveParamsSBS, object],
    sbs: SBSParams,
    bc: SBSBoundaryConditions,
) -> FWM_SBS_ModelParams:
    """
    Original bundler naming retained.
    """
    # Minimal runtime sanity checks (the RHS needs waves.omega)
    omega = np.asarray(getattr(waves, "omega"))
    if omega.shape != (4,):
        raise ValueError("waves.omega must have shape (4,)")
    if np.asarray(sbs.omega_B, dtype=float).shape != (4,):
        raise ValueError("sbs.omega_B must have shape (4,)")
    return FWM_SBS_ModelParams(fiber=fiber, waves=waves, sbs=sbs, bc=bc)
