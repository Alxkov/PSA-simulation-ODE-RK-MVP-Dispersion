"""
simulation.py

High-level simulation runner for the scalar Yaman-style 4-wave FWM model.

This modified version:
- treats omega1..omega4 as first-class inputs (wave order: pump1, pump2, signal, idler)
- computes (and caches) phase mismatch dbeta to be used inside exp(+- i dbeta z)
  using the centralized phase_matching layer
- supports dispersion-aware dbeta (Taylor / symmetric-even) as well as a PROVIDED dbeta
- supports legacy inputs (gamma/alpha in 1/(W·km), 1/km) via length_unit conversion

Important unit convention used internally in this runner:
- Integration coordinate z is in meters
- Therefore dbeta must be in 1/m inside exp(±i dbeta z)
- If you pass inputs in km-units, set length_unit="km" and this runner will convert.

Wave order across the project:
    [pump1, pump2, signal, idler]  ->  [omega1, omega2, omega3, omega4]
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence, Tuple

import numpy as np

import constants
from config import (
    SimulationConfig,
    custom_simulation_config,
    default_simulation_config,
    validate_config,
)
from dispersion import DispersionParams
from integrators import integrate_interval
from parameters import (
    FiberParams,
    PhaseMatchingParams,
    SimulationGrid,
    WavesParams,
    make_model_params,
)
from phase_matching import (
    PhaseMatchingConfig,
    PhaseMatchingMethod,
    PhaseMatchingResult,
    compute_phase_mismatch,
)
from yaman_model import rhs_yaman_simplified


# --------------------------------------------------------------------------------------
# Small helpers (units / validation / initial conditions)
# --------------------------------------------------------------------------------------

def _length_scale_to_m(length_unit: str) -> float:
    """
    Return scale factor to convert lengths in `length_unit` to meters.
    """
    u = str(length_unit).strip().lower()
    if u == "m":
        return 1.0
    if u == "km":
        return 1000.0
    raise ValueError(f"Unsupported length_unit={length_unit!r}. Use 'm' or 'km'.")


def _to_omega_array(omega: Sequence[float]) -> np.ndarray:
    om = np.asarray(list(omega), dtype=float)
    if om.shape != (4,):
        raise ValueError(f"omega must have shape (4,), got {om.shape}")
    if not np.all(np.isfinite(om)):
        raise ValueError("omega must be finite")
    if np.any(om <= 0.0):
        raise ValueError("omega must be positive (rad/s)")
    return om


def _to_power_array(p_in: Sequence[float]) -> np.ndarray:
    p = np.asarray(list(p_in), dtype=float)
    if p.shape != (4,):
        raise ValueError(f"p_in must have shape (4,), got {p.shape}")
    if not np.all(np.isfinite(p)):
        raise ValueError("p_in must be finite")
    if np.any(p < 0.0):
        raise ValueError("p_in must be non-negative (W)")
    return p


def _to_phase_array(phase_in: Optional[Sequence[float]]) -> np.ndarray:
    if phase_in is None:
        return np.zeros(4, dtype=float)
    ph = np.asarray(list(phase_in), dtype=float)
    if ph.shape != (4,):
        raise ValueError(f"phase_in must have shape (4,), got {ph.shape}")
    if not np.all(np.isfinite(ph)):
        raise ValueError("phase_in must be finite")
    return ph


def make_initial_amplitudes(
    p_in: Sequence[float],
    phase_in: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Construct complex amplitudes A0 at z=0 from powers and phases.

    Convention:
        |A_j|^2 = P_j,  A_j = sqrt(P_j) * exp(i * phi_j)

    Returns
    -------
    A0 : np.ndarray complex128, shape (4,)
    """
    p = _to_power_array(p_in)
    ph = _to_phase_array(phase_in)

    amp = np.sqrt(p).astype(np.complex128, copy=False)
    if np.any(ph != 0.0):
        amp *= np.exp(1j * ph)
    return amp


def _scale_dispersion_to_m(disp: DispersionParams, length_scale_to_m: float) -> DispersionParams:
    """
    Convert DispersionParams whose beta_n are expressed per `length_unit`
    into per-meter coefficients.

    If length_unit='km', length_scale_to_m=1000 and:
        beta_n [s^n/km] -> beta_n [s^n/m] by dividing by 1000.
    """
    s = float(length_scale_to_m)
    if s == 1.0:
        return disp

    extra_scaled = None
    if disp.extra is not None:
        extra_scaled = {int(k): float(v) / s for k, v in disp.extra.items()}

    return DispersionParams(
        omega_ref=disp.omega_ref,
        beta0=float(disp.beta0) / s,
        beta1=float(disp.beta1) / s,
        beta2=float(disp.beta2) / s,
        beta3=float(disp.beta3) / s,
        beta4=float(disp.beta4) / s,
        extra=extra_scaled,
    )


def _scale_phase_matching_cfg_to_m(cfg: PhaseMatchingConfig, length_scale_to_m: float) -> PhaseMatchingConfig:
    """
    If cfg.method == PROVIDED, interpret provided_delta_beta as per `length_unit`
    and convert it to 1/m.
    """
    if cfg.method != PhaseMatchingMethod.PROVIDED:
        return cfg

    if cfg.provided_delta_beta is None:
        raise ValueError("PhaseMatchingConfig.PROVIDED requires provided_delta_beta")

    s = float(length_scale_to_m)
    if s == 1.0:
        return cfg

    return PhaseMatchingConfig(
        method=PhaseMatchingMethod.PROVIDED,
        max_order=cfg.max_order,
        even_orders=cfg.even_orders,
        atol=cfg.atol,
        rtol=cfg.rtol,
        provided_delta_beta=float(cfg.provided_delta_beta) / s,
    )


def _default_phase_matching_cfg(
    *,
    dispersion: Optional[DispersionParams],
    beta_legacy: Optional[np.ndarray],
) -> PhaseMatchingConfig:
    """
    Choose a sensible default:
    - if dispersion is provided -> symmetric-even (2,4)
    - else if beta_legacy is provided -> PROVIDED using dbeta = beta3+beta4-beta1-beta2 (legacy style)
    - else -> error
    """
    if dispersion is not None:
        return PhaseMatchingConfig(
            method=PhaseMatchingMethod.SYMMETRIC_EVEN,
            max_order=4,
            even_orders=(2, 4),
            atol=0.0,
            rtol=1e-12,
            provided_delta_beta=None,
        )

    if beta_legacy is not None:
        b = np.asarray(beta_legacy, dtype=float)
        if b.shape != (4,):
            raise ValueError("beta_legacy must have shape (4,)")
        db = float((b[2] + b[3]) - (b[0] + b[1]))
        return PhaseMatchingConfig(
            method=PhaseMatchingMethod.PROVIDED,
            max_order=0,
            even_orders=(2,),
            atol=0.0,
            rtol=1e-12,
            provided_delta_beta=db,
        )

    raise ValueError("Provide either dispersion or beta_legacy (or an explicit phase_matching_cfg).")


# --------------------------------------------------------------------------------------
# Core simulation runner
# --------------------------------------------------------------------------------------

def run_single_simulation(
    cfg: SimulationConfig,
    *,
    gamma: float,
    alpha: float,
    omega: Sequence[float],
    p_in: Sequence[float],
    phase_in: Optional[Sequence[float]] = None,
    dispersion: Optional[DispersionParams] = None,
    phase_matching_cfg: Optional[PhaseMatchingConfig] = None,
    beta_legacy: Optional[Sequence[float]] = None,
    length_unit: str = "m",
    return_length_unit: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a single scalar 4-wave FWM simulation.

    Parameters
    ----------
    cfg:
        Numerical simulation configuration. cfg.z_max and cfg.dz are interpreted in `length_unit`.
    gamma:
        Kerr nonlinearity coefficient in units 1/(W·length_unit).
        Internally converted to 1/(W·m).
    alpha:
        Power attenuation coefficient in units 1/length_unit.
        Internally converted to 1/m.
    omega:
        Angular frequencies omega_j [rad/s], shape (4,), wave order [pump1,pump2,signal,idler].
    p_in:
        Input powers [W], shape (4,)
    phase_in:
        Input phases [rad], shape (4,) (optional)
    dispersion:
        DispersionParams with beta_n expressed per length_unit (i.e., s^n/length_unit).
        Internally converted to s^n/m.
        Required for dispersion-aware phase matching methods.
    phase_matching_cfg:
        Controls how dbeta is computed. If None, a default is chosen from dispersion/beta_legacy.
        If method == PROVIDED, provided_delta_beta is interpreted per length_unit and converted to 1/m.
    beta_legacy:
        Optional legacy beta(omega_j) values [1/length_unit] for backward compatibility.
        If you pass only beta_legacy (no dispersion), default dbeta method becomes PROVIDED with
        dbeta = beta3+beta4-beta1-beta2.
    length_unit:
        'm' or 'km'. Controls conversions to internal meters.
    return_length_unit:
        If None -> returns z in the same unit as `length_unit`.
        Otherwise 'm' or 'km' to force output units.

    Returns
    -------
    z_out:
        z-grid where solution is stored (in return_length_unit)
    A:
        Complex amplitudes, shape (N,4)
    """
    validate_config(cfg)

    scale_to_m = _length_scale_to_m(length_unit)

    # --- Inputs ---
    om = _to_omega_array(omega)
    p = _to_power_array(p_in)
    A0 = make_initial_amplitudes(p, phase_in)

    # --- Optional legacy betas ---
    beta_leg_m = None
    if beta_legacy is not None:
        b = np.asarray(list(beta_legacy), dtype=float)
        if b.shape != (4,):
            raise ValueError(f"beta_legacy must have shape (4,), got {b.shape}")
        if not np.all(np.isfinite(b)):
            raise ValueError("beta_legacy must be finite")
        # convert 1/length_unit -> 1/m
        beta_leg_m = b / scale_to_m

    # --- Dispersion conversion to per-meter ---
    disp_m = None
    if dispersion is not None:
        if not isinstance(dispersion, DispersionParams):
            raise TypeError("dispersion must be DispersionParams or None")
        disp_m = _scale_dispersion_to_m(dispersion, scale_to_m)

    # --- Phase matching config ---
    pm_cfg = phase_matching_cfg if phase_matching_cfg is not None else _default_phase_matching_cfg(
        dispersion=disp_m,
        beta_legacy=beta_leg_m,
    )
    if not isinstance(pm_cfg, PhaseMatchingConfig):
        raise TypeError("phase_matching_cfg must be PhaseMatchingConfig or None")

    pm_cfg = _scale_phase_matching_cfg_to_m(pm_cfg, scale_to_m)
    pm = PhaseMatchingParams(config=pm_cfg)

    # --- Build parameter containers (internal meters) ---
    fiber = FiberParams(
        length_m=float(cfg.z_max) * scale_to_m,
        gamma_W_m=float(gamma) / scale_to_m,
        alpha_1_m=float(alpha) / scale_to_m,
        dispersion=disp_m,
        beta_legacy_1_m=beta_leg_m,
    )

    waves = WavesParams(omega=om, symmetric=None)

    grid = SimulationGrid(
        dz_m=float(cfg.dz) * scale_to_m,
        z0_m=0.0,
    )

    params = make_model_params(
        waves=waves,
        fiber=fiber,
        grid=grid,
        phase_matching=pm,
    )

    # --- Compute and cache dbeta (in 1/m) for exp(±i dbeta z) ---
    # For PROVIDED, disp can be None; for other methods, disp must exist.
    res: PhaseMatchingResult = compute_phase_mismatch(
        omegas=params.waves.omega,
        disp=params.fiber.dispersion,
        cfg=params.phase_matching.config,
        symmetric_hint=params.waves.symmetric,
    )
    params.cache.set_phase_mismatch(res.delta_beta, symmetric=res.symmetric)

    # --- Run integration in meters ---
    z_m, A = integrate_interval(
        f=rhs_yaman_simplified,
        z_max=params.fiber.length_m,
        dz=params.grid.dz_m,
        y0=A0,
        params=params,
        save_every=cfg.save_every,
        check_nan=cfg.check_nan,
    )

    # --- Output unit conversion ---
    out_unit = length_unit if return_length_unit is None else return_length_unit
    out_scale_to_m = _length_scale_to_m(out_unit)
    z_out = z_m / out_scale_to_m

    return z_out, A


# --------------------------------------------------------------------------------------
# Example simulations (kept for quick sanity checks)
# --------------------------------------------------------------------------------------

def example_zero_signal() -> tuple[np.ndarray, np.ndarray]:
    """
    Example: two pumps, zero signal and idler at input, dbeta forced to 0 (PROVIDED).
    """
    cfg = default_simulation_config()

    length_unit = "km"  # matches the legacy config comments in your current project

    gamma = 1.3   # 1/(W·km)
    alpha = 0.0   # 1/km

    # Use proper angular frequency omega = 2πc/λ
    omega0 = 2.0 * np.pi * constants.c / 1.55e-6
    omega = np.array([omega0, omega0, omega0, omega0], dtype=float)

    p_in = np.array([0.5, 0.5, 0.0, 0.0], dtype=float)

    pm_cfg = PhaseMatchingConfig(
        method=PhaseMatchingMethod.PROVIDED,
        provided_delta_beta=0.0,  # interpreted as 1/km here and converted internally to 1/m
    )

    return run_single_simulation(
        cfg,
        gamma=gamma,
        alpha=alpha,
        omega=omega,
        p_in=p_in,
        phase_in=None,
        dispersion=None,
        phase_matching_cfg=pm_cfg,
        beta_legacy=None,
        length_unit=length_unit,
        return_length_unit=length_unit,
    )


def custom_seeded_signal() -> tuple[np.ndarray, np.ndarray]:
    """
    Example: seeded signal/idler with dbeta specified explicitly (PROVIDED).
    """
    cfg = custom_simulation_config(z_max=0.5, dz=1e-4)

    length_unit = "km"

    gamma = 10.0  # 1/(W·km)
    alpha = 0.0   # 1/km

    omega0 = 2.0 * np.pi * constants.c / 1.55e-6
    omega = np.array([omega0, omega0, omega0, omega0], dtype=float)

    P1 = 1e-1
    p_in = np.array([P1, P1, 1e-4, 1e-6], dtype=float)

    phase_in = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # Example: choose a mismatch in 1/km (will be converted to 1/m internally)
    ideal_mismatch = 0.0

    pm_cfg = PhaseMatchingConfig(
        method=PhaseMatchingMethod.PROVIDED,
        provided_delta_beta=float(ideal_mismatch),
    )

    return run_single_simulation(
        cfg,
        gamma=gamma,
        alpha=alpha,
        omega=omega,
        p_in=p_in,
        phase_in=phase_in,
        dispersion=None,
        phase_matching_cfg=pm_cfg,
        beta_legacy=None,
        length_unit=length_unit,
        return_length_unit=length_unit,
    )
