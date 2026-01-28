"""
fwm_sbs_model.py

Coupled Scalar FWM + SBS model (RHS functions):

A) Original "full" method (explicit RK4 on [A,Q]):
   - rhs_forward_AQ(z, y_AQ, params, B_at_z, ...) -> dy_AQ/dz
   - rhs_backward_B(z, B, params, A_at_z, Q_at_z) -> dB/dz

B) Exponential-Q method support (stiffness-friendly):
   - rhs_forward_A(z, A, params, B_at_z, Q_at_z, ...) -> dA/dz
   - q_exponential_step(Qn, A_used, B_used, dz, params, delta_omega, ...) -> Q_{n+1}
   - rhs_forward_Q(z, Q, params, A_at_z, B_at_z, ...) -> dQ/dz   (for diagnostics)


Conventions assumed:
- z in km
- gamma in 1/(W·km)
- beta in 1/km (only used for Δβ in FWM)
- vA in km/s
- GammaB in 1/s
- DeltaOmega in rad/s
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from parameters_sbs import FWM_SBS_ModelParams


# -----------------------------------------------------------------------------
# Shape / dtype helpers
# -----------------------------------------------------------------------------

def _as_complex_len4(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128, copy=False)
    return arr


def _as_complex_len8(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.shape != (8,):
        raise ValueError(f"{name} must have shape (8,), got {arr.shape}")
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128, copy=False)
    return arr


def unpack_AQ(y_AQ: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_AQ = _as_complex_len8(y_AQ, "y_AQ")
    return y_AQ[:4], y_AQ[4:]


def pack_AQ(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    A = _as_complex_len4(A, "A")
    Q = _as_complex_len4(Q, "Q")
    return np.concatenate([A, Q]).astype(np.complex128, copy=False)


# -----------------------------------------------------------------------------
# Phase mismatch and detuning
# -----------------------------------------------------------------------------

def compute_dbeta(betas: np.ndarray) -> float:
    """
    Δβ = β3 + β4 - β1 - β2
    betas must be float array shape (4,) = [β1, β2, β3, β4].
    """
    b = np.asarray(betas, dtype=float)
    if b.shape != (4,):
        raise ValueError(f"betas must have shape (4,), got {b.shape}")
    return float(b[2] + b[3] - b[0] - b[1])


def compute_delta_omega(params: FWM_SBS_ModelParams) -> np.ndarray:
    """
    ΔΩ_j = ΩB - (ωAj - ωBj)
    where ωAj = params.waves.omega[j], ωBj = params.sbs.omega_B[j]
    returns float array shape (4,)
    """
    omega_A = np.asarray(params.waves.omega, dtype=float)
    omega_B = np.asarray(params.sbs.omega_B, dtype=float)
    if omega_A.shape != (4,):
        raise ValueError(f"params.waves.omega must have shape (4,), got {omega_A.shape}")
    if omega_B.shape != (4,):
        raise ValueError(f"params.sbs.omega_B must have shape (4,), got {omega_B.shape}")
    return float(params.sbs.OmegaB) - (omega_A - omega_B)


# -----------------------------------------------------------------------------
# Kerr prefactors
# -----------------------------------------------------------------------------

def kerr_prefactor_forward(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    For Aj equations:
      Kerr_j = Pj + 2*(sum_{k!=j}Pk) + 2*(sum_k Sk)
             = 2*(sumP + sumS) - Pj
    """
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    if P.shape != (4,) or S.shape != (4,):
        raise ValueError("P and S must have shape (4,)")
    sumP = float(np.sum(P))
    sumS = float(np.sum(S))
    return (2.0 * (sumP + sumS) - P).astype(float)


def kerr_prefactor_backward(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    For Bj equations appearing inside iγ(...)Bj on RHS of -dB/dz:
      KerrB_j = Sj + 2*(sum_k Pk) + 2*(sum_{k!=j}Sk)
              = 2*(sumP + sumS) - Sj
    """
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    if P.shape != (4,) or S.shape != (4,):
        raise ValueError("P and S must have shape (4,)")
    sumP = float(np.sum(P))
    sumS = float(np.sum(S))
    return (2.0 * (sumP + sumS) - S).astype(float)


# -----------------------------------------------------------------------------
# FWM mixing terms
# -----------------------------------------------------------------------------

def fwm_terms(z: float, A: np.ndarray, dbeta: float) -> np.ndarray:
    """
    Returns only the mixing products (WITHOUT the prefactor i*gamma).
    You multiply it by (1j*gamma*2) in the caller.

    Ordering: A = [A1,A2,A3,A4]
    """
    A = _as_complex_len4(A, "A")
    A1, A2, A3, A4 = A

    phase_pumps = np.exp(1j * float(dbeta) * float(z))    # +iΔβz
    phase_side  = np.exp(-1j * float(dbeta) * float(z))   # -iΔβz

    term_A1 = phase_pumps * (np.conj(A2) * A3 * A4)
    term_A2 = phase_pumps * (np.conj(A1) * A3 * A4)
    term_A3 = phase_side  * (np.conj(A4) * A1 * A2)
    term_A4 = phase_side  * (np.conj(A3) * A1 * A2)

    return np.array([term_A1, term_A2, term_A3, term_A4], dtype=np.complex128)


# -----------------------------------------------------------------------------
# NEW: Pieces for exponential-Q method (and for cleaner structure in general)
# -----------------------------------------------------------------------------

def rhs_forward_A(
    z: float,
    A: np.ndarray,
    params: FWM_SBS_ModelParams,
    B_at_z: np.ndarray,
    Q_at_z: np.ndarray,
    *,
    dbeta: Optional[float] = None,
) -> np.ndarray:
    """
    Forward optical RHS for A only (shape (4,)).

    dA/dz = iγ( Kerr(A,B)*A + 2*FWM_mix(A) ) + i*kappa1 * B * Q
    """
    A = _as_complex_len4(A, "A")
    B = _as_complex_len4(B_at_z, "B_at_z")
    Q = _as_complex_len4(Q_at_z, "Q_at_z")

    gamma = float(params.fiber.gamma)
    kappa1 = float(params.sbs.kappa1)

    # Powers
    P = np.abs(A) ** 2
    S = np.abs(B) ** 2

    # Kerr contribution
    kA = kerr_prefactor_forward(P, S)
    dA_kerr = 1j * gamma * (kA * A)

    # FWM contribution
    if dbeta is None:
        betas = np.asarray(params.fiber.beta, dtype=float)
        if betas.shape != (4,):
            raise ValueError(f"params.fiber.beta must have shape (4,), got {betas.shape}")
        db = compute_dbeta(betas)
    else:
        db = float(dbeta)

    mix = fwm_terms(float(z), A, db)
    dA_fwm = 1j * gamma * 2.0 * mix

    # SBS coupling
    dA_sbs = 1j * kappa1 * (B * Q)

    return (dA_kerr + dA_fwm + dA_sbs).astype(np.complex128, copy=False)


def rhs_forward_Q(
    z: float,
    Q: np.ndarray,
    params: FWM_SBS_ModelParams,
    A_at_z: np.ndarray,
    B_at_z: np.ndarray,
    *,
    delta_omega: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Acoustic RHS for Q only (shape (4,)).

    vA dQ/dz = -(GammaB/2 + iΔΩ) Q + i*kappa2 * A * conj(B)

    => dQ/dz = [-(GammaB/2 + iΔΩ) Q + i*kappa2 * A * conj(B)] / vA
    """
    Q = _as_complex_len4(Q, "Q")
    A = _as_complex_len4(A_at_z, "A_at_z")
    B = _as_complex_len4(B_at_z, "B_at_z")

    kappa2 = float(params.sbs.kappa2)
    GammaB = float(params.sbs.GammaB)
    vA = float(params.sbs.vA)
    if vA <= 0.0:
        raise ValueError(f"params.sbs.vA must be > 0 (km/s), got {vA}")

    if delta_omega is None:
        dOm = compute_delta_omega(params)
    else:
        dOm = np.asarray(delta_omega, dtype=float)
        if dOm.shape != (4,):
            raise ValueError(f"delta_omega must have shape (4,), got {dOm.shape}")

    damping = (GammaB / 2.0) + 1j * dOm
    dQ = (-(damping * Q) + 1j * kappa2 * (A * np.conj(B))) / vA
    return dQ.astype(np.complex128, copy=False)


def q_lambda(params: FWM_SBS_ModelParams, delta_omega: np.ndarray) -> np.ndarray:
    """
    lambda_j = (GammaB/2 + i*DeltaOmega_j) / vA     [1/km]
    """
    GammaB = float(params.sbs.GammaB)
    vA = float(params.sbs.vA)
    if vA <= 0.0:
        raise ValueError(f"params.sbs.vA must be > 0 (km/s), got {vA}")

    dOm = np.asarray(delta_omega, dtype=float)
    if dOm.shape != (4,):
        raise ValueError(f"delta_omega must have shape (4,), got {dOm.shape}")

    return (GammaB / 2.0 + 1j * dOm) / vA


def q_forcing(params: FWM_SBS_ModelParams, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    F_j = (i*kappa2/vA) * A_j * conj(B_j)
    """
    kappa2 = float(params.sbs.kappa2)
    vA = float(params.sbs.vA)
    if vA <= 0.0:
        raise ValueError(f"params.sbs.vA must be > 0 (km/s), got {vA}")

    A = _as_complex_len4(A, "A")
    B = _as_complex_len4(B, "B")
    return (1j * kappa2 * (A * np.conj(B))) / vA


def q_exponential_step(
    Qn: np.ndarray,
    A_used: np.ndarray,
    B_used: np.ndarray,
    dz: float,
    params: FWM_SBS_ModelParams,
    *,
    delta_omega: Optional[np.ndarray] = None,
    lambda_eps: float = 1e-14,
) -> np.ndarray:
    """
    Exponential (exact-linear) update for Q over one spatial step dz:

      dQ/dz = -lambda Q + F

    with
      lambda = (GammaB/2 + iΔΩ)/vA
      F      = (i*kappa2/vA) A conj(B)

    Exact-with-constant-forcing step:
      Q_{n+1} = exp(-lambda dz) Q_n + (1 - exp(-lambda dz))/lambda * F_used

    Uses expm1 for numerical stability:
      (1 - exp(-x))/x = -expm1(-x)/x

    If |lambda| is extremely small, falls back to Q_{n+1} ≈ Q_n + dz * F.
    """
    Qn = _as_complex_len4(Qn, "Qn")
    A_used = _as_complex_len4(A_used, "A_used")
    B_used = _as_complex_len4(B_used, "B_used")

    dz = float(dz)
    if dz == 0.0:
        return Qn.copy()

    if delta_omega is None:
        dOm = compute_delta_omega(params)
    else:
        dOm = np.asarray(delta_omega, dtype=float)
        if dOm.shape != (4,):
            raise ValueError(f"delta_omega must have shape (4,), got {dOm.shape}")

    lam = q_lambda(params, dOm)          # (4,) complex
    F = q_forcing(params, A_used, B_used)

    E = np.exp(-lam * dz)

    G = np.empty((4,), dtype=np.complex128)
    for j in range(4):
        if abs(lam[j]) < float(lambda_eps):
            G[j] = dz
        else:
            G[j] = -np.expm1(-lam[j] * dz) / lam[j]

    return (E * Qn + G * F).astype(np.complex128, copy=False)


# -----------------------------------------------------------------------------
# Original full forward RHS retained (now built from the pieces above)
# -----------------------------------------------------------------------------

def rhs_forward_AQ(
    z: float,
    y_AQ: np.ndarray,
    params: FWM_SBS_ModelParams,
    B_at_z: np.ndarray,
    *,
    dbeta: Optional[float] = None,
    delta_omega: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Original forward RHS for y_AQ = [A,Q] (shape (8,)).

    This remains available for the "full" algorithm (explicit RK4 on [A,Q]).
    """
    y_AQ = _as_complex_len8(y_AQ, "y_AQ")
    B = _as_complex_len4(B_at_z, "B_at_z")

    A, Q = unpack_AQ(y_AQ)

    dA = rhs_forward_A(
        z=z,
        A=A,
        params=params,
        B_at_z=B,
        Q_at_z=Q,
        dbeta=dbeta,
    )

    dQ = rhs_forward_Q(
        z=z,
        Q=Q,
        params=params,
        A_at_z=A,
        B_at_z=B,
        delta_omega=delta_omega,
    )

    return pack_AQ(dA, dQ)


# -----------------------------------------------------------------------------
# Backward RHS retained (unchanged interface)
# -----------------------------------------------------------------------------

def rhs_backward_B(
    z: float,
    B: np.ndarray,
    params: FWM_SBS_ModelParams,
    A_at_z: np.ndarray,
    Q_at_z: np.ndarray,
) -> np.ndarray:
    """
    Backward optical RHS for B (shape (4,)).

    Equation form:
      -dB/dz = iγ( KerrB * B ) + i*kappa1 * A * Q*

    Therefore (standard derivative):
      dB/dz = -iγ( KerrB * B ) - i*kappa1 * A * Q*
    """
    # z is kept for signature consistency; not explicitly used here.
    B = _as_complex_len4(B, "B")
    A = _as_complex_len4(A_at_z, "A_at_z")
    Q = _as_complex_len4(Q_at_z, "Q_at_z")

    gamma = float(params.fiber.gamma)
    kappa1 = float(params.sbs.kappa1)

    P = np.abs(A) ** 2
    S = np.abs(B) ** 2
    kB = kerr_prefactor_backward(P, S)

    dB = (-1j * gamma * (kB * B)) + (-1j * kappa1 * (A * np.conj(Q)))
    return dB.astype(np.complex128, copy=False)
