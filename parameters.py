"""
parameters.py

Centralized parameter dataclasses for the scalar FWM project (dual-pump / 4-wave model).

This version is modified to:
- make frequencies (omega1..omega4) first-class (wave order: pump1, pump2, signal, idler)
- attach dispersion parameters (Taylor betan around omega_ref, typically omegac)
- attach a phase-matching configuration
- provide a cache slot for the computed phase mismatch dbeta used in exp(±i dbeta z)

Units (recommended throughout the project):
- Length: meters (m)
- omega: rad/s
- beta: 1/m ; betan: s^n / m
- gamma: 1/(W·m)
- alpha: 1/m  (power attenuation coefficient; if you use field attenuation, be consistent elsewhere)

Dependencies:
- numpy
- frequency_plan.py (builders / SymmetricPlan)
- dispersion.py (DispersionParams)
- phase_matching.py (PhaseMatchingConfig, PhaseMatchingMethod)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from dispersion import DispersionParams
from frequency_plan import (
    SymmetricPlan,
    plan_from_omegas,
    plan_from_symmetry,
    plan_from_wavelengths,
)
from phase_matching import PhaseMatchingConfig, PhaseMatchingMethod


WAVE_ORDER: Tuple[str, str, str, str] = ("pump1", "pump2", "signal", "idler")


def _as_omega_array(omegas: Sequence[float], *, name: str = "omegas") -> np.ndarray:
    arr = np.asarray(list(omegas), dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain positive angular frequencies (rad/s)")
    return arr


def _as_beta_array(betas: Sequence[float], *, name: str = "betas") -> np.ndarray:
    arr = np.asarray(list(betas), dtype=float)
    if arr.shape != (4,):
        raise ValueError(f"{name} must have shape (4,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    return arr


def _to_float(x: float, *, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise TypeError(f"{name} must be a real scalar, got {type(x)!r}") from e
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def _validate_nonneg(x: float, *, name: str) -> float:
    v = _to_float(x, name=name)
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0, got {v!r}")
    return v


def _validate_positive(x: float, *, name: str) -> float:
    v = _to_float(x, name=name)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


@dataclass(frozen=True, slots=True)
class WavesParams:
    """
    Optical wave frequency plan.

    Attributes
    ----------
    omega : np.ndarray shape (4,)
        [omega1, omega2, omega3, omega4] in wave order [pump1, pump2, signal, idler]
    symmetric : SymmetricPlan | None
        Optional (omegac, omegad, omega) representation consistent with omega.
        If not provided, it can be inferred elsewhere (e.g., phase_matching).
    """
    omega: np.ndarray
    symmetric: Optional[SymmetricPlan] = None

    def __post_init__(self) -> None:
        om = _as_omega_array(self.omega, name="omega")
        object.__setattr__(self, "omega", om)

        if self.symmetric is not None:
            if not isinstance(self.symmetric, SymmetricPlan):
                raise TypeError("symmetric must be SymmetricPlan or None")
            # ensure the symmetric plan generates the same omega within tolerance
            om_sym = self.symmetric.omegas()
            if not np.allclose(om, om_sym, rtol=1e-12, atol=0.0):
                raise ValueError(
                    "Provided symmetric plan is inconsistent with omega. "
                    f"omega={om}, omega(sym)={om_sym}"
                )

    @property
    def omega1(self) -> float:
        return float(self.omega[0])

    @property
    def omega2(self) -> float:
        return float(self.omega[1])

    @property
    def omega3(self) -> float:
        return float(self.omega[2])

    @property
    def omega4(self) -> float:
        return float(self.omega[3])

    @classmethod
    def from_symmetry(cls, omega_c: float, omega_d: float, Omega: float) -> "WavesParams":
        sp = SymmetricPlan(omega_c=omega_c, omega_d=omega_d, Omega=Omega)
        return cls(omega=sp.omegas(), symmetric=sp)

    @classmethod
    def from_omegas(
        cls,
        omega1: float,
        omega2: float,
        omega3: float,
        omega4: Optional[float] = None,
    ) -> "WavesParams":
        om = plan_from_omegas(omega1, omega2, omega3, omega4)
        # No symmetric cached by default (can be inferred cheaply later)
        return cls(omega=om, symmetric=None)

    @classmethod
    def from_wavelengths(
        cls,
        lambda1_m: float,
        lambda2_m: float,
        lambda3_m: float,
        lambda4_m: Optional[float] = None,
    ) -> "WavesParams":
        om = plan_from_wavelengths(lambda1_m, lambda2_m, lambda3_m, lambda4_m)
        return cls(omega=om, symmetric=None)


@dataclass(frozen=True, slots=True)
class FiberParams:
    """
    Fiber / waveguide parameters for scalar FWM.

    Attributes
    ----------
    length_m : float
        Propagation length [m]
    gamma_W_m : float
        Nonlinear coefficient gamma [1/(W·m)]
    alpha_1_m : float
        Power attenuation coefficient alpha [1/m] (set 0 if lossless)
    dispersion : DispersionParams | None
        Taylor dispersion model around omega_ref (typically omegac).
        Required for dispersion-aware phase matching methods.
    beta_legacy_1_m : np.ndarray | None
        Optional legacy per-wave beta(omegaj) values [1/m] in wave order.
        Kept ONLY for backward compatibility; do not use in new dispersion workflow.
    """
    length_m: float
    gamma_W_m: float
    alpha_1_m: float = 0.0
    dispersion: Optional[DispersionParams] = None
    beta_legacy_1_m: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        L = _validate_positive(self.length_m, name="length_m")
        g = _to_float(self.gamma_W_m, name="gamma_W_m")
        a = _validate_nonneg(self.alpha_1_m, name="alpha_1_m")

        object.__setattr__(self, "length_m", L)
        object.__setattr__(self, "gamma_W_m", g)
        object.__setattr__(self, "alpha_1_m", a)

        if self.dispersion is not None and not isinstance(self.dispersion, DispersionParams):
            raise TypeError("dispersion must be DispersionParams or None")

        if self.beta_legacy_1_m is not None:
            bl = _as_beta_array(self.beta_legacy_1_m, name="beta_legacy_1_m")
            object.__setattr__(self, "beta_legacy_1_m", bl)


@dataclass(frozen=True, slots=True)
class SimulationGrid:
    """
    Discretization parameters for ODE integration.
    """
    dz_m: float
    z0_m: float = 0.0

    def __post_init__(self) -> None:
        dz = _validate_positive(self.dz_m, name="dz_m")
        z0 = _to_float(self.z0_m, name="z0_m")
        object.__setattr__(self, "dz_m", dz)
        object.__setattr__(self, "z0_m", z0)


@dataclass(frozen=True, slots=True)
class PhaseMatchingParams:
    """
    How dbeta is computed.
    """
    config: PhaseMatchingConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, PhaseMatchingConfig):
            raise TypeError("config must be a PhaseMatchingConfig")


@dataclass(slots=True)
class CacheParams:
    """
    Runtime cache populated at simulation start (or on-demand).

    Notes:
    - Keep this mutable: it is filled after parameters are constructed.
    - RHS should use cache.delta_beta in exp(±i dbeta z) for dispersion-aware runs.
    """
    delta_beta_1_m: Optional[float] = None
    symmetric: Optional[SymmetricPlan] = None

    def set_phase_mismatch(self, delta_beta_1_m: float, symmetric: Optional[SymmetricPlan] = None) -> None:
        db = _to_float(delta_beta_1_m, name="delta_beta_1_m")
        self.delta_beta_1_m = db
        self.symmetric = symmetric


@dataclass(frozen=True, slots=True)
class ModelParams:
    """
    Aggregated model parameters passed into integrators / RHS.
    """
    waves: WavesParams
    fiber: FiberParams
    grid: SimulationGrid
    phase_matching: PhaseMatchingParams
    cache: CacheParams

    def __post_init__(self) -> None:
        if not isinstance(self.cache, CacheParams):
            raise TypeError("cache must be a CacheParams (mutable cache object)")


def make_default_phase_matching_params(*, method: PhaseMatchingMethod = PhaseMatchingMethod.SYMMETRIC_EVEN) -> PhaseMatchingParams:
    """
    Helper: default dbeta strategy for dispersion-aware runs.
    """
    cfg = PhaseMatchingConfig(method=method, max_order=4, even_orders=(2, 4), atol=0.0, rtol=1e-12)
    return PhaseMatchingParams(config=cfg)


def make_model_params(
    *,
    waves: WavesParams,
    fiber: FiberParams,
    grid: SimulationGrid,
    phase_matching: Optional[PhaseMatchingParams] = None,
) -> ModelParams:
    """
    Factory that also initializes an empty cache.

    You are expected to compute dbeta once at simulation start and store it in params.cache
    (unless you use method == PROVIDED, etc.).
    """
    pm = phase_matching if phase_matching is not None else make_default_phase_matching_params()
    cache = CacheParams(delta_beta_1_m=None, symmetric=waves.symmetric)
    return ModelParams(waves=waves, fiber=fiber, grid=grid, phase_matching=pm, cache=cache)
