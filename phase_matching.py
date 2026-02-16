"""
phase_matching.py

Phase-matching / phase-mismatch (dbeta) computation layer for scalar FWM simulations.

This module centralizes how dbeta is computed so the ODE/RHS code does not need to know
about dispersion modeling choices.

Wave order convention across the project:
    [pump1, pump2, signal, idler]  ->  [omega1, omega2, omega3, omega4]

Definitions (as in your equations sheet):
    omegac = (omega1 + omega2)/2
    omegad = (omega1 - omega2)/2
    omega  = omega3 - omegac

Then:
    omega1 = omegac + omegad
    omega2 = omegac - omegad
    omega3 = omegac + omega
    omega4 = omegac - omega

Phase mismatch:
    dbeta = beta(omega3) + beta(omega4) - beta(omega1) - beta(omega2)

Provided strategies:
- "general_taylor": compute beta(omega) via Taylor and assemble dbeta from omega1..omega4
- "symmetric_even": use the symmetric even-order closed form
                  dbeta = sum_{n even>=2} beta_n(omegac) * (omega^n - omegad^n) * 2/n!
  which yields dbeta ≈ beta2(omega^2-omegad^2) + beta4/12(omega^4-omegad^4) + ...
- "provided": use a user-provided constant dbeta (useful for debugging / legacy)

Dependencies:
- frequency_plan.py
- dispersion.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from dispersion import DispersionParams, delta_beta_from_omegas, delta_beta_symmetric
from frequency_plan import SymmetricPlan, infer_symmetry_from_omegas


class PhaseMatchingMethod(str, Enum):
    GENERAL_TAYLOR = "general_taylor"
    SYMMETRIC_EVEN = "symmetric_even"
    PROVIDED = "provided"


def _to_scalar_float(x: float, *, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise TypeError(f"{name} must be a real scalar, got {type(x)!r}") from e
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def _as_omega_array(omegas: Sequence[float], *, name: str = "omegas") -> np.ndarray:
    arr = np.asarray(list(omegas), dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain only positive angular frequencies (rad/s)")
    return arr


@dataclass(frozen=True)
class PhaseMatchingConfig:
    """
    Configuration for dbeta computation.

    Parameters
    ----------
    method:
        - "general_taylor": dbeta from beta(omegaj) Taylor model
        - "symmetric_even": dbeta from symmetric even-order closed form
        - "provided": use provided_delta_beta

    max_order:
        Highest Taylor order for "general_taylor" (e.g., 2, 3, 4).

    even_orders:
        Even orders to include for "symmetric_even" (e.g., (2,4) or (2,4,6)).

    atol, rtol:
        Tolerances for energy conservation checks when assembling from omega’s.
    """
    method: PhaseMatchingMethod = PhaseMatchingMethod.SYMMETRIC_EVEN
    max_order: int = 4
    even_orders: Tuple[int, ...] = (2, 4)
    atol: float = 0.0
    rtol: float = 1e-12

    # Used only if method == PROVIDED
    provided_delta_beta: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, PhaseMatchingMethod):
            # allow string input
            try:
                object.__setattr__(self, "method", PhaseMatchingMethod(str(self.method)))
            except Exception as e:
                raise ValueError(f"Invalid method {self.method!r}") from e

        if not isinstance(self.max_order, int) or self.max_order < 0:
            raise ValueError(f"max_order must be int >= 0, got {self.max_order!r}")

        ev = tuple(self.even_orders)
        if len(ev) == 0:
            raise ValueError("even_orders must not be empty (e.g., (2,4))")
        for n in ev:
            if not isinstance(n, int):
                raise TypeError("even_orders must contain ints")
            if n < 2 or (n % 2) != 0:
                raise ValueError(f"even_orders must contain even ints >= 2, got {n!r}")

        a = _to_scalar_float(self.atol, name="atol")
        r = _to_scalar_float(self.rtol, name="rtol")
        if a < 0.0 or r < 0.0:
            raise ValueError("atol and rtol must be >= 0")
        object.__setattr__(self, "atol", a)
        object.__setattr__(self, "rtol", r)

        if self.method == PhaseMatchingMethod.PROVIDED:
            if self.provided_delta_beta is None:
                raise ValueError("provided_delta_beta must be set when method == 'provided'")
            pdb = _to_scalar_float(self.provided_delta_beta, name="provided_delta_beta")
            object.__setattr__(self, "provided_delta_beta", pdb)


@dataclass(frozen=True)
class PhaseMatchingResult:
    """
    Returned by compute_phase_mismatch: contains dbeta and (optionally) the symmetric variables.
    """
    delta_beta: float
    symmetric: Optional[SymmetricPlan] = None


def compute_phase_mismatch(
    omegas: Sequence[float],
    disp: Optional[DispersionParams],
    cfg: PhaseMatchingConfig,
    *,
    symmetric_hint: Optional[SymmetricPlan] = None,
) -> PhaseMatchingResult:
    """
    Compute dbeta for the given omega1..omega4.

    Parameters
    ----------
    omegas:
        Sequence of 4 angular frequencies [rad/s] in order [omega1,omega2,omega3,omega4].
    disp:
        Dispersion parameters (required unless cfg.method == PROVIDED).
        For SYMMETRIC_EVEN, disp.omega_ref should ideally equal omegac.
    cfg:
        Phase matching configuration.
    symmetric_hint:
        Optional SymmetricPlan if we have omegac, omegad, omega. If not provided,
        it will be inferred from omega1, omega2, omega3 (and omega4 checked).

    Returns
    -------
    PhaseMatchingResult(delta_beta, symmetric)
    """
    om = _as_omega_array(omegas, name="omegas")

    if cfg.method == PhaseMatchingMethod.PROVIDED:
        return PhaseMatchingResult(delta_beta=float(cfg.provided_delta_beta), symmetric=None)

    if disp is None:
        raise ValueError("disp must be provided unless method == 'provided'")

    if cfg.method == PhaseMatchingMethod.GENERAL_TAYLOR:
        db = delta_beta_from_omegas(
            om,
            disp,
            max_order=cfg.max_order,
            atol=cfg.atol,
            rtol=cfg.rtol,
        )
        return PhaseMatchingResult(delta_beta=float(db), symmetric=None)

    if cfg.method == PhaseMatchingMethod.SYMMETRIC_EVEN:
        sp = symmetric_hint
        if sp is None:
            sp = infer_symmetry_from_omegas(
                omega1=float(om[0]),
                omega2=float(om[1]),
                omega3=float(om[2]),
                omega4=float(om[3]),
                atol=cfg.atol,
                rtol=cfg.rtol,
            )
        db = delta_beta_symmetric(
            omega_c=sp.omega_c,
            omega_d=sp.omega_d,
            Omega=sp.Omega,
            disp=disp,
            even_orders=cfg.even_orders,
        )
        return PhaseMatchingResult(delta_beta=float(db), symmetric=sp)

    raise ValueError(f"Unsupported phase-matching method: {cfg.method!r}")


@dataclass(frozen=True)
class PhaseMismatchCalculator:
    """
    Convenience callable object to compute dbeta repeatedly with a fixed config/dispersion.

    Example
    -------
    calc = PhaseMismatchCalculator(disp=disp, cfg=cfg)
    res = calc(omegas)
    dbeta = res.delta_beta
    """
    disp: Optional[DispersionParams]
    cfg: PhaseMatchingConfig

    def __call__(
        self,
        omegas: Sequence[float],
        *,
        symmetric_hint: Optional[SymmetricPlan] = None,
    ) -> PhaseMatchingResult:
        return compute_phase_mismatch(
            omegas=omegas,
            disp=self.disp,
            cfg=self.cfg,
            symmetric_hint=symmetric_hint,
        )
