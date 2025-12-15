import numpy as np
from numpy.ma.core import conjugate

from parameters import ModelParams

def rhs_yaman_simplified(
        z: float,
        a_arr: np.ndarray,
        params: ModelParams,
) -> np.ndarray:
    """
    Right-hand side (RHS) of the simplified Yaman ODE system (Eq. 2.63–2.64) in z-domain.

    The solver integrates:
        dA/dz = rhs_yaman_simplified(z, A, p)
    """

    # --- Basic validation (kept lightweight for performance) ---
    a_arr = np.asarray(a_arr)

    if a_arr.shape != (4,):
        raise ValueError("A must have shape (4,)")
    if not np.iscomplexobj(a_arr):
        # Enforce complex dtype to avoid subtle casting bugs
        a_arr = a_arr.astype(np.complex128, copy=False)

    a1, a2, a3, a4 = a_arr
    # --- Extract frequently used parameters ---
    gamma = float(params.fiber.gamma)   # [1/(W·m)]
    alpha = float(params.fiber.alpha)   # [1/m]
    beta = np.asarray(params.fiber.beta, dtype=float)   # [1/m], shape (4,)

    if beta.shape != (4,):
        raise ValueError("p.fiber.beta must have shape (4,)")

def _linear_terms_stub(a_arr: np.ndarray, alpha: float, beta: np.ndarray) -> np.ndarray:
    """
    Stub for linear terms.
    """

    return np.zeros_like(a_arr)


def _kerr_terms_stub(a_arr: np.ndarray, gamma: float) -> np.ndarray:
    """
    Stub for Kerr SPM/XPM terms.

    Intended structure (example):
        P = |A|^2
        dA_j/dz += 1j*gamma*(SPM_j + XPM_j)*A_j

    """

    pass
