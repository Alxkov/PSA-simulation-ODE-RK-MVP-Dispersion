"""
frequency_plan.py

Frequency plan utilities for 4-wave mixing (FWM).

Wave order across the project is assumed to be:
    [pump1, pump2, signal, idler]  ->  [omega1, omega2, omega3, omega4]

This module provides:
- Conversions between wavelength lambda [m], frequency f [Hz], and angular frequency omega [rad/s]
- Builders for omega1..omega4 from:
    (a) explicit omegas
    (b) wavelengths
    (c) symmetric parameters (omegac, omegad, omega) used:
        omegac = (omega1 + omega2)/2
        omegad = (omega1 - omega2)/2
        omega  = omega3 - omegac
        omega1 = omegac + omegad
        omega2 = omegac - omegad
        omega3 = omegac + omega
        omega4 = omegac - omega

Energy conservation (for FWM):
    omega1 + omega2 = omega3 + omega4
so given (omega1, omega2, omega3) we infer omega4 = omega1 + omega2 - omega3.

Dependencies:
    - numpy
    - constants.c  (speed of light in vacuum) [m/s]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

import constants


_TWO_PI = 2.0 * np.pi
_WAVE_ORDER = ("pump1", "pump2", "signal", "idler")


def _to_scalar_float(x: float, *, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise TypeError(f"{name} must be a real scalar, got {type(x)!r}") from e
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def _validate_omega_positive(omega: float, *, name: str) -> float:
    w = _to_scalar_float(omega, name=name)
    if w <= 0.0:
        raise ValueError(f"{name} must be > 0 (rad/s), got {w!r}")
    return w


def _validate_lambda_positive(lambda_m: float, *, name: str) -> float:
    lam = _to_scalar_float(lambda_m, name=name)
    if lam <= 0.0:
        raise ValueError(f"{name} must be > 0 (m), got {lam!r}")
    return lam


def _validate_f_positive(f_hz: float, *, name: str) -> float:
    f = _to_scalar_float(f_hz, name=name)
    if f <= 0.0:
        raise ValueError(f"{name} must be > 0 (Hz), got {f!r}")
    return f


def omega_from_f(f_hz: float) -> float:
    """Convert frequency f [Hz] to angular frequency omega [rad/s]."""
    f = _validate_f_positive(f_hz, name="f_hz")
    return _TWO_PI * f


def f_from_omega(omega: float) -> float:
    """Convert angular frequency omega [rad/s] to frequency f [Hz]."""
    w = _validate_omega_positive(omega, name="omega")
    return w / _TWO_PI


def omega_from_lambda(lambda_m: float) -> float:
    """Convert vacuum wavelength lambda [m] to angular frequency omega [rad/s] using omega = 2πc/lambda."""
    lam = _validate_lambda_positive(lambda_m, name="lambda_m")
    return _TWO_PI * constants.c / lam


def lambda_from_omega(omega: float) -> float:
    """Convert angular frequency omega [rad/s] to vacuum wavelength lambda [m] using lambda = 2πc/omega."""
    w = _validate_omega_positive(omega, name="omega")
    return _TWO_PI * constants.c / w


def _as_omega_array(omegas: Iterable[float], *, name: str = "omega") -> np.ndarray:
    arr = np.asarray(list(omegas), dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain only positive angular frequencies (rad/s)")
    return arr


def enforce_energy_conservation(
    omega: np.ndarray,
    *,
    atol: float = 0.0,
    rtol: float = 1e-12,
) -> None:
    """
    Validate omega1 + omega2 == omega3 + omega4 within tolerance.

    Raises ValueError if violated.
    """
    om = _as_omega_array(omega, name="omega")
    lhs = om[0] + om[1]
    rhs = om[2] + om[3]
    if not np.isclose(lhs, rhs, atol=atol, rtol=rtol):
        diff = lhs - rhs
        raise ValueError(
            "Energy conservation violated: omega1+omega2 != omega3+omega4. "
            f"(lhs={lhs:.16e}, rhs={rhs:.16e}, diff={diff:.16e})"
        )


@dataclass(frozen=True)
class SymmetricPlan:
    """
    Symmetric frequency plan parameters.

    Definitions:
        omegac = (omega1 + omega2)/2
        omegad = (omega1 - omega2)/2
        omega  = omega3 - omegac

    Units: rad/s
    """
    omega_c: float  # omegac
    omega_d: float  # omegad
    Omega: float    # omega

    def __post_init__(self) -> None:
        oc = _validate_omega_positive(self.omega_c, name="omega_c")
        od = _to_scalar_float(self.omega_d, name="omega_d")
        Om = _to_scalar_float(self.Omega, name="Omega")

        # Not strictly required by math, but catches nonsense early:
        # If |omegad| >= omegac then one of the pumps would have omega <= 0.
        if abs(od) >= oc:
            raise ValueError(
                f"Invalid symmetric plan: |omega_d| must be < omega_c to keep omega1, omega2 positive. "
                f"Got omega_c={oc!r}, omega_d={od!r}"
            )

        # Ensure stored values are the validated scalar floats
        object.__setattr__(self, "omega_c", oc)
        object.__setattr__(self, "omega_d", od)
        object.__setattr__(self, "Omega", Om)

    @property
    def omega1(self) -> float:
        return self.omega_c + self.omega_d

    @property
    def omega2(self) -> float:
        return self.omega_c - self.omega_d

    @property
    def omega3(self) -> float:
        return self.omega_c + self.Omega

    @property
    def omega4(self) -> float:
        return self.omega_c - self.Omega

    def omegas(self) -> np.ndarray:
        """
        Return omega array in project wave order:
            [omega1, omega2, omega3, omega4] = [pump1, pump2, signal, idler]
        """
        om = np.array([self.omega1, self.omega2, self.omega3, self.omega4], dtype=float)
        # omega1, omega2 are guaranteed positive by __post_init__;
        # omega3, omega4 positivity depends on omega relative to omegac.
        if np.any(om <= 0.0):
            raise ValueError(
                "This symmetric plan produces non-positive omega for signal/idler. "
                f"Computed omega=[{om[0]:.6e}, {om[1]:.6e}, {om[2]:.6e}, {om[3]:.6e}] rad/s. "
                "Adjust Omega and/or omega_c."
            )
        enforce_energy_conservation(om)
        return om


def plan_from_symmetry(omega_c: float, omega_d: float, Omega: float) -> np.ndarray:
    """
    Build omega = [omega1, omega2, omega3, omega4] from symmetric parameters (omegac, omegad, omega).

    Returns
    -------
    np.ndarray shape (4,)
        [pump1, pump2, signal, idler] angular frequencies [rad/s]
    """
    sp = SymmetricPlan(omega_c=omega_c, omega_d=omega_d, Omega=Omega)
    return sp.omegas()


def infer_symmetry_from_omegas(
    omega1: float,
    omega2: float,
    omega3: float,
    omega4: Optional[float] = None,
    *,
    atol: float = 0.0,
    rtol: float = 1e-12,
) -> SymmetricPlan:
    """
    Infer (omegac, omegad, omega) from omega1, omega2, omega3 (and optional omega4).

    If omega4 is provided, validates energy conservation.
    If omega4 is None, infers omega4 = omega1 + omega2 - omega3.
    """
    w1 = _validate_omega_positive(omega1, name="omega1")
    w2 = _validate_omega_positive(omega2, name="omega2")
    w3 = _validate_omega_positive(omega3, name="omega3")

    if omega4 is None:
        w4 = w1 + w2 - w3
        w4 = _validate_omega_positive(w4, name="omega4(inferred)")
    else:
        w4 = _validate_omega_positive(omega4, name="omega4")
        om = np.array([w1, w2, w3, w4], dtype=float)
        enforce_energy_conservation(om, atol=atol, rtol=rtol)

    omega_c = 0.5 * (w1 + w2)
    omega_d = 0.5 * (w1 - w2)
    Omega = w3 - omega_c
    sp = SymmetricPlan(omega_c=omega_c, omega_d=omega_d, Omega=Omega)

    # Final consistency check against provided/inferred omega4:
    om_sp = sp.omegas()
    if not np.isclose(om_sp[3], w4, atol=atol, rtol=rtol):
        raise ValueError(
            "Inferred symmetric parameters are inconsistent with omega4. "
            f"omega4(target)={w4:.16e}, omega4(from symmetry)={om_sp[3]:.16e}"
        )

    return sp


def plan_from_omegas(
    omega1: float,
    omega2: float,
    omega3: float,
    omega4: Optional[float] = None,
    *,
    atol: float = 0.0,
    rtol: float = 1e-12,
) -> np.ndarray:
    """
    Build omega array from omega1, omega2, omega3 (and optional omega4).

    If omega4 is None:
        omega4 = omega1 + omega2 - omega3   (energy conservation)

    If omega4 is provided:
        validates omega1+omega2 == omega3+omega4 within tolerance.
    """
    w1 = _validate_omega_positive(omega1, name="omega1")
    w2 = _validate_omega_positive(omega2, name="omega2")
    w3 = _validate_omega_positive(omega3, name="omega3")

    if omega4 is None:
        w4 = w1 + w2 - w3
        w4 = _validate_omega_positive(w4, name="omega4(inferred)")
    else:
        w4 = _validate_omega_positive(omega4, name="omega4")

    om = np.array([w1, w2, w3, w4], dtype=float)
    enforce_energy_conservation(om, atol=atol, rtol=rtol)
    return om


def plan_from_wavelengths(
    lambda1_m: float,
    lambda2_m: float,
    lambda3_m: float,
    lambda4_m: Optional[float] = None,
    *,
    atol: float = 0.0,
    rtol: float = 1e-12,
) -> np.ndarray:
    """
    Build omega array from vacuum wavelengths (meters).

    If lambda4_m is None:
        w4 = w1 + w2 - w3 (energy conservation)  -> then lambda4 = (2 pi c)/omega4

    Note:
        Energy conservation is exact in omega-space. Doing it in lambda-space is incorrect.
        So we always convert lambda -> omega, apply conservation, then optionally validate lambda4.
    """
    lam1 = _validate_lambda_positive(lambda1_m, name="lambda1_m")
    lam2 = _validate_lambda_positive(lambda2_m, name="lambda2_m")
    lam3 = _validate_lambda_positive(lambda3_m, name="lambda3_m")

    w1 = omega_from_lambda(lam1)
    w2 = omega_from_lambda(lam2)
    w3 = omega_from_lambda(lam3)

    if lambda4_m is None:
        w4 = w1 + w2 - w3
        w4 = _validate_omega_positive(w4, name="omega4(inferred)")
    else:
        lam4 = _validate_lambda_positive(lambda4_m, name="lambda4_m")
        w4 = omega_from_lambda(lam4)

    om = np.array([w1, w2, w3, w4], dtype=float)
    enforce_energy_conservation(om, atol=atol, rtol=rtol)
    return om


def describe_plan(omega: np.ndarray) -> str:
    """
    Human-readable description of a plan: omega and corresponding lambda, f.

    Returns a multi-line string (no printing).
    """
    om = _as_omega_array(omega, name="omega")
    lam = np.array([lambda_from_omega(w) for w in om], dtype=float)
    f = np.array([f_from_omega(w) for w in om], dtype=float)

    lines = []
    lines.append("Frequency plan (wave order: pump1, pump2, signal, idler):")
    for i, label in enumerate(_WAVE_ORDER):
        lines.append(
            f"  {label:6s}: "
            f"omega={om[i]: .16e} rad/s, "
            f"f={f[i]: .16e} Hz, "
            f"lambda={lam[i]: .16e} m"
        )
    lines.append(f"  Check: omega1+omega2 - (omega3+omega4) = {(om[0]+om[1])-(om[2]+om[3]): .16e} rad/s")
    return "\n".join(lines)
