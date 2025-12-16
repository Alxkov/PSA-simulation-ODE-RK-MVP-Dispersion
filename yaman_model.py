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

    Parameters
    ----------
    z : float
        Propagation coordinate. Units: km.
    a_arr : np.ndarray
        Complex amplitudes, shape (4,). Units: sqrt(W).
    params : ModelParams
        Model parameters (fiber, dispersion, nonlinearity).

    Returns
    -------
    np.ndarray
        dz-derivative dA/dz, complex array of shape (4,).

    Raises
    ------
    ValueError
        If `a_arr` does not have shape (4,) or `beta` does not have shape (4,).
    """

    a_arr = np.asarray(a_arr)
    if a_arr.shape != (4,):
        raise ValueError("a_arr must have shape (4,)")

    if not np.iscomplexobj(a_arr):
        a_arr = a_arr.astype(np.complex128, copy=False)

    gamma = float(params.fiber.gamma)  # [1/(W·m)]
    alpha = float(params.fiber.alpha)  # [1/m]
    beta = np.asarray(params.fiber.beta, dtype=float)  # [1/m], shape (4,)

    if beta.shape != (4,):
        raise ValueError("params.fiber.beta must have shape (4,)")

    linear = _linear_terms_stub(z, a_arr, alpha, beta)
    kerr = _kerr_terms_stub(z, a_arr, gamma)
    fwm = _fwm_terms_stub(z, a_arr, alpha, beta)

    return linear + kerr + fwm



def _linear_terms_stub(z: float, a_arr: np.ndarray, alpha: float, beta: np.ndarray) -> np.ndarray:
    """
    Stub for linear terms.
    """

    return np.zeros_like(a_arr)


def _kerr_terms_stub(z: float, a_arr: np.ndarray, gamma: float) -> np.ndarray:
    """
    Stub for Kerr SPM/XPM terms.

    Intended structure (example):
        P = |A|^2
        dA_j/dz += 1j*gamma*(SPM_j + XPM_j)*A_j

    """

    two_thirds = 2.0 / 3.0
    four_thirds = 4.0 / 3.0

    powers = np.abs(a_arr) ** 2
    total_power = powers.sum()
    other_power = total_power - powers

    factor = two_thirds * powers + four_thirds * other_power
    return 1j * gamma * factor * a_arr[0]


import numpy as np

def _fwm_terms_stub(
    z: float,
    a_arr: np.ndarray,
    _alpha: float,
    beta: np.ndarray,
) -> np.ndarray:
    """
    Stub for Four-Wave Mixing (FWM) terms.

    Parameters
    ----------
    z : float
        Propagation coordinate.
    a_arr : np.ndarray
        Complex amplitudes, expected shape (4,).
    _alpha : float
        Unused in this stub (kept for API compatibility).
    beta : np.ndarray
        Propagation constants, expected shape (4,), ordered as [0, 1, 2, 3].

    Returns
    -------
    np.ndarray
        Complex array of the same shape as `a_arr`.
    """
    four_thirds = 4.0 / 3.0
    dbeta = beta[2] + beta[3] - beta[0] - beta[1]
    phase = np.exp(-1j * dbeta * z)

    result = np.empty_like(a_arr, dtype=np.complex128)

    for i in range(len(a_arr)):
        product = 1.0 + 0.0j
        for j, aj in enumerate(a_arr):
            if j == i:
                continue

            same_group = (i < 2) == (j < 2)  # {0,1} vs {2,3}
            product *= np.conj(aj) if same_group else aj

        result[i] = four_thirds * phase * product

    return result

