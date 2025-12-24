import numpy as np

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

    gamma = float(params.fiber.gamma)  # [1/(W·km)]
    alpha = float(params.fiber.alpha)  # [1/km]
    betas = np.asarray(params.fiber.beta, dtype=float)  # [1/km], shape (4,)

    if betas.shape != (4,):
        raise ValueError("params.fiber.beta must have shape (4,)")

    linear = _linear_terms_stub(z, a_arr, alpha, betas)
    kerr = _kerr_terms_stub(z, a_arr, gamma)
    fwm = _fwm_terms_stub(z, a_arr, alpha, betas, gamma)

    return linear + kerr + fwm


def _linear_terms_stub(z: float, a_arr: np.ndarray, alpha: float, betas: np.ndarray) -> np.ndarray:
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

    one = 1.0
    two = 2.0

    pump1, pump2, signal, idler = a_arr

    p_p1 = np.abs(pump1) ** 2
    p_p2 = np.abs(pump2) ** 2
    p_s = np.abs(signal) ** 2
    p_i = np.abs(idler) ** 2

    f_p1 = one * p_p1 + two * (p_p2 + p_s + p_i)
    f_p2 = one * p_p2 + two * (p_p1 + p_s + p_i)
    f_s = one * p_s + two * (p_p1 + p_p2 + p_i)
    f_i = one * p_i + two * (p_p1 + p_p2 + p_s)

    return 1j * gamma * np.array(
        [f_p1 * pump1, f_p2 * pump2, f_s * signal, f_i * idler],
        dtype=np.complex128,
    )


def _fwm_terms_stub(
    z: float,
    a_arr: np.ndarray,
    _alpha: float,
    betas: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Stub for Four-Wave Mixing (FWM) terms using explicit wave names.

    Wave order in a_arr:
        a_arr[0] = pump1
        a_arr[1] = pump2
        a_arr[2] = signal
        a_arr[3] = idler
    """
    two = 2.0

    pump1, pump2, signal, idler = a_arr
    beta_p1, beta_p2, beta_s, beta_i = betas

    dbeta = beta_s + beta_i - beta_p1 - beta_p2

    phase_pumps = np.exp(1j * dbeta * z)
    phase_sidebands = np.exp(-1j * dbeta * z)

    term_pump1 = phase_pumps * (np.conj(pump2) * signal * idler)
    term_pump2 = phase_pumps * (np.conj(pump1) * signal * idler)

    term_signal = phase_sidebands * (pump1 * pump2 * np.conj(idler))
    term_idler  = phase_sidebands * (pump1 * pump2 * np.conj(signal))

    return 1j * gamma * two * np.array(
        [term_pump1, term_pump2, term_signal, term_idler],
        dtype=np.complex128,
    )
