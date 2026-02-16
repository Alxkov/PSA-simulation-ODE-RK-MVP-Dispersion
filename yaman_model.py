# yaman_model.py

from __future__ import annotations

import numpy as np

from parameters import ModelParams


def rhs_yaman_simplified(
    z: float,
    a_arr: np.ndarray,
    params: ModelParams,
) -> np.ndarray:
    """
    RHS of the scalar 4-wave FWM ODE system (Yaman-style / your dispersion sheet).

    Wave order:
        a_arr = [A1, A2, A3, A4] = [pump1, pump2, signal, idler]

    Model (no linear phase terms; only SPM/XPM + FWM mixing):
        dA1/dz =  i gamma( (P1 + 2(P2+P3+P4))A1 + 2 A2* A3 A4 exp(+idbeta z) )
        dA2/dz =  i gamma( (P2 + 2(P1+P3+P4))A2 + 2 A1* A3 A4 exp(+idbeta z) )
        dA3/dz =  i gamma( (P3 + 2(P1+P2+P4))A3 + 2 A4* A1 A2 exp(-idbeta z) )
        dA4/dz =  i gamma( (P4 + 2(P1+P2+P3))A4 + 2 A3* A1 A2 exp(-idbeta z) )

    Optional loss (if provided in params): field attenuation term -alpha/2 * A_j.

    IMPORTANT:
        This implementation expects dbeta to be already computed by the phase-matching layer
        and stored in params.cache.delta_beta_1_m (preferred).
        For backward compatibility, it can fall back to params.fiber.beta or beta_legacy_1_m.

    Units consistency:
        z and dbeta must be in reciprocal units (e.g., z[m] with dbeta[1/m], or z[km] with dbeta[1/km]).
    """
    a_arr = np.asarray(a_arr)
    if a_arr.shape != (4,):
        raise ValueError("a_arr must have shape (4,)")

    if not np.iscomplexobj(a_arr):
        a_arr = a_arr.astype(np.complex128, copy=False)
    else:
        a_arr = a_arr.astype(np.complex128, copy=False)

    gamma, alpha, dbeta = _extract_gamma_alpha_dbeta(params)

    linear = _linear_loss_terms(a_arr, alpha)
    kerr = _kerr_terms(a_arr, gamma)
    fwm = _fwm_terms(z, a_arr, gamma, dbeta)

    return linear + kerr + fwm


# --------------------------------------------------------------------------------------
# Parameter extraction (supports new + legacy parameter containers)
# --------------------------------------------------------------------------------------

def _extract_gamma_alpha_dbeta(params: ModelParams) -> tuple[float, float, float]:
    """
    Returns (gamma, alpha, dbeta) in the SAME length units used for z.

    Priority:
    - gamma: fiber.gamma_W_m (new) else fiber.gamma (legacy)
    - alpha: fiber.alpha_1_m (new) else fiber.alpha (legacy) else 0
    - dbeta: cache.delta_beta_1_m (new) else
             fiber.beta_legacy_1_m (new optional) else fiber.beta (legacy)
    """
    if not hasattr(params, "fiber"):
        raise ValueError("params must have attribute 'fiber'")

    fiber = params.fiber

    # gamma
    if hasattr(fiber, "gamma_W_m"):
        gamma = float(fiber.gamma_W_m)
    elif hasattr(fiber, "gamma"):
        gamma = float(fiber.gamma)
    else:
        raise ValueError("Fiber parameters must contain gamma_W_m (new) or gamma (legacy).")

    # alpha (optional)
    if hasattr(fiber, "alpha_1_m"):
        alpha = float(fiber.alpha_1_m)
    elif hasattr(fiber, "alpha"):
        alpha = float(fiber.alpha)
    else:
        alpha = 0.0

    # dbeta
    dbeta = None
    if hasattr(params, "cache") and params.cache is not None:
        if hasattr(params.cache, "delta_beta_1_m"):
            dbeta = params.cache.delta_beta_1_m

    if dbeta is None:
        if hasattr(fiber, "beta_legacy_1_m") and fiber.beta_legacy_1_m is not None:
            betas = np.asarray(fiber.beta_legacy_1_m, dtype=float)
        elif hasattr(fiber, "beta") and fiber.beta is not None:
            betas = np.asarray(fiber.beta, dtype=float)
        else:
            raise ValueError(
                "Phase mismatch dbeta is not available. "
                "Expected params.cache.delta_beta_1_m to be set (preferred), "
                "or fiber.beta_legacy_1_m / fiber.beta to exist for fallback."
            )

        if betas.shape != (4,):
            raise ValueError("Fallback betas must have shape (4,)")

        # dbeta = beta3 + beta4 - beta1 - beta2 (signal+idler - pump1 - pump2)
        dbeta = float((betas[2] + betas[3]) - (betas[0] + betas[1]))

    dbeta = float(dbeta)

    return gamma, alpha, dbeta


# --------------------------------------------------------------------------------------
# Physics terms
# --------------------------------------------------------------------------------------

def _linear_loss_terms(a_arr: np.ndarray, alpha: float) -> np.ndarray:
    """
    Optional attenuation in the field domain:
        dA/dz += -(alpha/2) A

    Here alpha is assumed to be the *power* attenuation coefficient.
    """
    if alpha == 0.0:
        return np.zeros_like(a_arr)
    return (-0.5 * alpha) * a_arr


def _kerr_terms(a_arr: np.ndarray, gamma: float) -> np.ndarray:
    """
    Kerr SPM/XPM terms (scalar, co-polarized assumption):

        dA_j/dz += i gamma ( |A_j|^2 + 2 Σ_{k≠j} |A_k|^2 ) A_j
    """
    pump1, pump2, signal, idler = a_arr

    p_p1 = np.abs(pump1) ** 2
    p_p2 = np.abs(pump2) ** 2
    p_s = np.abs(signal) ** 2
    p_i = np.abs(idler) ** 2

    f_p1 = p_p1 + 2.0 * (p_p2 + p_s + p_i)
    f_p2 = p_p2 + 2.0 * (p_p1 + p_s + p_i)
    f_s = p_s + 2.0 * (p_p1 + p_p2 + p_i)
    f_i = p_i + 2.0 * (p_p1 + p_p2 + p_s)

    return (1j * gamma) * np.array(
        [f_p1 * pump1, f_p2 * pump2, f_s * signal, f_i * idler],
        dtype=np.complex128,
    )


def _fwm_terms(z: float, a_arr: np.ndarray, gamma: float, dbeta: float) -> np.ndarray:
    """
    Four-wave mixing terms with explicit phase mismatch in the exponential:

        Pumps:      exp(+i dbeta z)
        Sidebands:  exp(-i dbeta z)

    Terms:
        dA1/dz += igamma * 2 * ( A2* A3 A4 exp(+idbetaz) )
        dA2/dz += igamma * 2 * ( A1* A3 A4 exp(+idbetaz) )
        dA3/dz += igamma * 2 * ( A4* A1 A2 exp(-idbetaz) )
        dA4/dz += igamma * 2 * ( A3* A1 A2 exp(-idbetaz) )
    """
    pump1, pump2, signal, idler = a_arr

    phase_pumps = np.exp(1j * dbeta * z)
    phase_sidebands = np.exp(-1j * dbeta * z)

    term_pump1 = phase_pumps * (np.conj(pump2) * signal * idler)
    term_pump2 = phase_pumps * (np.conj(pump1) * signal * idler)

    term_signal = phase_sidebands * (np.conj(idler) * pump1 * pump2)
    term_idler = phase_sidebands * (np.conj(signal) * pump1 * pump2)

    return (1j * gamma * 2.0) * np.array(
        [term_pump1, term_pump2, term_signal, term_idler],
        dtype=np.complex128,
    )
