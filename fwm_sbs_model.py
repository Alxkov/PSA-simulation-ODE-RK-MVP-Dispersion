"""
fwm_sbs_model.py

RHS implementations for the coupled Scalar FWM + SBS ODE system given in
ScalarFWM__SBS.pdf (Eqs. 1–12):

Forward waves (A1..A4):
  dAj/dz = iγ( Kerr_j * Aj + FWM_j ) + i*kappa1 * Bj * Qj

Backward waves (B1..B4):
  -dBj/dz = iγ( KerrB_j * Bj ) + i*kappa1 * Aj * Qj*

Acoustic waves (Q1..Q4):
  vA * dQj/dz = -(ΓB/2 + iΔΩj) Qj + i*kappa2 * Aj * Bj*

Simplification used here:
- kappa1 and kappa2 are scalars, identical for all j=1..4.

Conventions assumed (consistent with your current project):
- z is in km
- γ is in 1/(W·km)
- β are in 1/km (used only in Δβ inside the FWM exponent)
- vA is in km/s
- ΓB is in 1/s
- ΔΩ are in rad/s

Then (ΓB/2 + iΔΩ)/vA has units 1/km, consistent with dQ/dz.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from parameters_sbs import FWM_SBS_ModelParams


# ---------------------------------------------------------------------
# Basic helpers / validation
# ---------------------------------------------------------------------

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
    """
    Split stacked state y_AQ = [A1..A4, Q1..Q4] into (A, Q).
    """
    y_AQ = _as_complex_len8(y_AQ, "y_AQ")
    A = y_AQ[:4]
    Q = y_AQ[4:]
    return A, Q


def pack_AQ(A: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Stack A and Q into y_AQ = [A1..A4, Q1..Q4].
    """
    A = _as_complex_len4(A, "A")
    Q = _as_complex_len4(Q, "Q")
    return np.concatenate([A, Q]).astype(np.complex128, copy=False)


# ---------------------------------------------------------------------
# Phase-mismatch and detuning
# ---------------------------------------------------------------------

def compute_dbeta(betas: np.ndarray) -> float:
    """
    Δβ = β(ω3) + β(ω4) - β(ω1) - β(ω2)
    betas must be a float array of length 4: [β1, β2, β3, β4]
    """
    b = np.asarray(betas, dtype=float)
    if b.shape != (4,):
        raise ValueError(f"betas must have shape (4,), got {b.shape}")
    return float(b[2] + b[3] - b[0] - b[1])


def compute_delta_omega(params: FWM_SBS_ModelParams) -> np.ndarray:
    """
    ΔΩj = ΩB - (ωAj - ωBj), j=1..4
    Uses:
      ωAj = params.waves.omega[j]
      ωBj = params.sbs.omega_B[j]
    """
    omega_A = np.asarray(params.waves.omega, dtype=float)
    omega_B = np.asarray(params.sbs.omega_B, dtype=float)
    if omega_A.shape != (4,):
        raise ValueError(f"params.waves.omega must have shape (4,), got {omega_A.shape}")
    if omega_B.shape != (4,):
        raise ValueError(f"params.sbs.omega_B must have shape (4,), got {omega_B.shape}")
    return float(params.sbs.OmegaB) - (omega_A - omega_B)


# ---------------------------------------------------------------------
# Kerr and FWM terms
# ---------------------------------------------------------------------

def kerr_prefactor_forward(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    For Aj equations:
      Kerr_j = Pj + 2*(sum_{k!=j} Pk) + 2*(sum_k Sk)
            = 2*(sumP + sumS) - Pj
    Returns array length 4.
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
    For Bj equations (appearing inside iγ(...)Bj on the RHS of -dBj/dz):
      KerrB_j = Sj + 2*(sum_k Pk) + 2*(sum_{k!=j} Sk)
             = 2*(sumP + sumS) - Sj
    Returns array length 4.
    """
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    if P.shape != (4,) or S.shape != (4,):
        raise ValueError("P and S must have shape (4,)")
    sumP = float(np.sum(P))
    sumS = float(np.sum(S))
    return (2.0 * (sumP + sumS) - S).astype(float)


def fwm_terms(z: float, A: np.ndarray, dbeta: float, gamma: float) -> np.ndarray:
    """
    FWM part in Aj equations (the '2 A* A A exp(±iΔβ z)' terms).

    Wave order:
      A[0]=A1, A[1]=A2, A[2]=A3, A[3]=A4

    Returns ONLY the iγ*(2*...) contribution (i.e., already multiplied by iγ*2).
    """
    A = _as_complex_len4(A, "A")

    A1, A2, A3, A4 = A

    phase_pumps = np.exp(1j * dbeta * z)     # +iΔβz
    phase_side  = np.exp(-1j * dbeta * z)    # -iΔβz

    # Eq (1)-(4) FWM mixing terms:
    # dA1/dz ... + iγ*(2 * A2* A3 A4 e^{+iΔβz})
    # dA2/dz ... + iγ*(2 * A1* A3 A4 e^{+iΔβz})
    # dA3/dz ... + iγ*(2 * A4* A1 A2 e^{-iΔβz})
    # dA4/dz ... + iγ*(2 * A3* A1 A2 e^{-iΔβz})

    term_A1 = phase_pumps * (np.conj(A2) * A3 * A4)
    term_A2 = phase_pumps * (np.conj(A1) * A3 * A4)
    term_A3 = phase_side  * (np.conj(A4) * A1 * A2)
    term_A4 = phase_side  * (np.conj(A3) * A1 * A2)

    return 1j * float(gamma) * 2.0 * np.array([term_A1, term_A2, term_A3, term_A4], dtype=np.complex128)


# ---------------------------------------------------------------------
# Main RHS functions
# ---------------------------------------------------------------------

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
    RHS for forward sweep state y_AQ = [A1..A4, Q1..Q4], given B(z).

    Parameters
    ----------
    z : float
        Propagation coordinate (km).
    y_AQ : np.ndarray
        Complex state, shape (8,) = [A1..A4, Q1..Q4].
    params : FWM_SBS_ModelParams
        Contains fiber (gamma, beta), waves (omega), sbs (kappa1,kappa2,GammaB,vA,omega_B).
    B_at_z : np.ndarray
        Backward fields at the same z, shape (4,) complex.
    dbeta : float, optional
        If provided, uses this dBeta; otherwise computes from params.fiber.beta.
    delta_omega : np.ndarray, optional
        If provided, uses this dOmega array (shape (4,)); otherwise computes from params.

    Returns
    -------
    np.ndarray
        dy_AQ/dz, shape (8,) complex.
    """
    y_AQ = _as_complex_len8(y_AQ, "y_AQ")
    B = _as_complex_len4(B_at_z, "B_at_z")
    A, Q = unpack_AQ(y_AQ)

    gamma = float(params.fiber.gamma)
    betas = np.asarray(params.fiber.beta, dtype=float)
    if betas.shape != (4,):
        raise ValueError(f"params.fiber.beta must have shape (4,), got {betas.shape}")

    kappa1 = float(params.sbs.kappa1)
    kappa2 = float(params.sbs.kappa2)
    GammaB = float(params.sbs.GammaB)
    vA = float(params.sbs.vA)

    if vA <= 0.0:
        raise ValueError(f"params.sbs.vA must be > 0 (km/s), got {vA}")

    # Powers
    P = np.abs(A) ** 2
    S = np.abs(B) ** 2

    # Kerr (Aj equations)
    kA = kerr_prefactor_forward(P, S)
    dA_kerr = 1j * gamma * (kA * A)

    # FWM
    if dbeta is None:
        db = compute_dbeta(betas)
    else:
        db = float(dbeta)
    dA_fwm = fwm_terms(z, A, db, gamma)

    # SBS coupling in Aj equations: + i*kappa1 * Bj * Qj
    dA_sbs = 1j * kappa1 * (B * Q)

    dA = dA_kerr + dA_fwm + dA_sbs

    # Acoustic detunings
    if delta_omega is None:
        dOm = compute_delta_omega(params)
    else:
        dOm = np.asarray(delta_omega, dtype=float)
        if dOm.shape != (4,):
            raise ValueError(f"delta_omega must have shape (4,), got {dOm.shape}")

    # Q equations:
    # vA dQ/dz = -(GammaB/2 + iΔΩ) Q + i*kappa2 * A * B*
    # => dQ/dz = [-(GammaB/2 + iΔΩ) Q + i*kappa2 * A * conj(B)] / vA
    damping = (GammaB / 2.0) + 1j * dOm
    dQ = (-(damping * Q) + 1j * kappa2 * (A * np.conj(B))) / vA

    return pack_AQ(dA, dQ)


def rhs_backward_B(
    z: float,
    B: np.ndarray,
    params: FWM_SBS_ModelParams,
    A_at_z: np.ndarray,
    Q_at_z: np.ndarray,
) -> np.ndarray:
    """
    RHS for backward sweep of B(z), given A(z) and Q(z) at the same z.

    Implements Eq (5)-(8):
      -dBj/dz = iγ( KerrB_j * Bj ) + i*kappa1 * Aj * Qj*

    We return dB/dz (standard derivative w.r.t. z):
      dBj/dz = -iγ( KerrB_j * Bj ) - i*kappa1 * Aj * Qj*

    Notes
    -----
    In the solver you will typically integrate B from z=L to z=0 (decreasing z).
    Your RK4 integrator can handle this if you pass a reversed z-grid (so dz < 0).

    Parameters
    ----------
    z : float
        Propagation coordinate (km). (Not explicitly used here, but kept for RHS signature.)
    B : np.ndarray
        Backward fields, shape (4,) complex.
    params : FWM_SBS_ModelParams
        Model parameters.
    A_at_z : np.ndarray
        Forward fields at same z, shape (4,) complex.
    Q_at_z : np.ndarray
        Acoustic fields at same z, shape (4,) complex.

    Returns
    -------
    np.ndarray
        dB/dz, shape (4,) complex.
    """
    B = _as_complex_len4(B, "B")
    A = _as_complex_len4(A_at_z, "A_at_z")
    Q = _as_complex_len4(Q_at_z, "Q_at_z")

    gamma = float(params.fiber.gamma)
    kappa1 = float(params.sbs.kappa1)

    # Powers
    P = np.abs(A) ** 2
    S = np.abs(B) ** 2

    # Kerr prefactor inside iγ(...)Bj on the RHS of -dB/dz
    kB = kerr_prefactor_backward(P, S)

    # dB/dz = -iγ*kB*B - i*kappa1*A*Q*
    dB = (-1j * gamma * (kB * B)) + (-1j * kappa1 * (A * np.conj(Q)))

    return dB
