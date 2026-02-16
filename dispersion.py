"""
dispersion.py

Dispersion utilities for scalar FWM simulations.

Core responsibilities:
- Represent a Taylor expansion of the propagation constant beta(omega) around a reference omega_ref (typically omega_c).
- Compute beta(omega) using that Taylor model.
- Compute phase mismatch:
      dbeta = beta(omega3) + beta(omega4) - beta(omega1) - beta(omega2)
  either directly from omega1..omega4 or using the symmetric (omega_c, omega_d, omega) variables:
      omega1 = omega_c + omega_d,  omega2 = omega_c - omega_d,  omega3 = omega_c + omega,  omega4 = omega_c - omega
  which yields (even-order-only):
      dbeta ≈ beta2(omega_c)(omega^2 - omega_d^2) + beta4(omega_c)/12 (omega^4 - omega_d^4) + ...

Also includes helpers to convert fiber dispersion parameters D and S into beta2 and beta3:

    beta2 = -(lambda^2 / (2pic)) D
    beta3 = (lambda^4 / (4pi^2 c^2)) ( S + 2D/lambda )

Notes on units:
- omega in rad/s
- beta in 1/m
- beta1 in s/m
- beta2 in s^2/m
- beta3 in s^3/m
- beta4 in s^4/m
- D in s/m^2   (often given as ps/(nm·km))
- S in s/m^3   (often given as ps/(nm^2·km))

Dependencies:
    - numpy
    - constants.c  (speed of light in vacuum) [m/s]
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

import constants


_TWO_PI = 2.0 * np.pi


def _to_scalar_float(x: float, *, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise TypeError(f"{name} must be a real scalar, got {type(x)!r}") from e
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return v


def _validate_positive(x: float, *, name: str) -> float:
    v = _to_scalar_float(x, name=name)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


def _omega_from_lambda(lambda_m: float) -> float:
    lam = _validate_positive(lambda_m, name="lambda_m")
    return _TWO_PI * constants.c / lam


def D_ps_nm_km_to_SI(D_ps_nm_km: float) -> float:
    """
    Convert D from ps/(nm*km) to SI units s/m^2.

    1 ps/(nm·km) = 1e-12 s / (1e-9 m * 1e3 m) = 1e-6 s/m^2
    """
    D = _to_scalar_float(D_ps_nm_km, name="D_ps_nm_km")
    return D * 1e-6


def S_ps_nm2_km_to_SI(S_ps_nm2_km: float) -> float:
    """
    Convert S from ps/(nm^2*km) to SI units s/m^3.

    1 ps/(nm^2·km) = 1e-12 s / (1e-18 m^2 * 1e3 m) = 1e3 s/m^3
    """
    S = _to_scalar_float(S_ps_nm2_km, name="S_ps_nm2_km")
    return S * 1e3


def beta2_from_D(lambda_ref_m: float, D_SI: float) -> float:
    """
    Compute beta2 [s^2/m] from dispersion parameter D [s/m^2] at wavelength lambda.

        beta2 = -(lambda^2 / (2pic)) D
    """
    lam = _validate_positive(lambda_ref_m, name="lambda_ref_m")
    D = _to_scalar_float(D_SI, name="D_SI")
    return -((lam * lam) / (_TWO_PI * constants.c)) * D


def beta3_from_D_S(lambda_ref_m: float, D_SI: float, S_SI: float) -> float:
    """
    Compute beta3 [s^3/m] from D [s/m^2] and slope S [s/m^3] at wavelength lambda.

        beta3 = (lambda^4 / (4pi^2 c^2)) * ( S + 2D/lambda )
    """
    lam = _validate_positive(lambda_ref_m, name="lambda_ref_m")
    D = _to_scalar_float(D_SI, name="D_SI")
    S = _to_scalar_float(S_SI, name="S_SI")

    pref = (lam**4) / ((2.0 * np.pi)**2 * constants.c**2)  # lambda^4 / (4pi^2 c^2)
    return pref * (S + 2.0 * D / lam)


@dataclass(frozen=True)
class DispersionParams:
    """
    Taylor expansion of beta(omega) around omega_ref:

        beta(omega) = beta0
             + beta1 domega
             + (beta2/2!) domega^2
             + (beta3/3!) domega^3
             + (beta4/4!) domega^4
             + ...

    Provide any subset; missing coefficients are treated as 0.

    Units:
        omega_ref : rad/s
        beta0    : 1/m
        beta1    : s/m
        beta2    : s^2/m
        beta3    : s^3/m
        beta4    : s^4/m
        (etc.)
    """
    omega_ref: float
    beta0: float = 0.0
    beta1: float = 0.0
    beta2: float = 0.0
    beta3: float = 0.0
    beta4: float = 0.0
    # Optional higher orders (even/odd) in a dict: {n: beta_n}
    # If provided, they override betaN fields for n>=5 (and may also include 0..4 ).
    extra: Optional[Dict[int, float]] = None

    def __post_init__(self) -> None:
        wref = _validate_positive(self.omega_ref, name="omega_ref")
        object.__setattr__(self, "omega_ref", wref)

        for name in ("beta0", "beta1", "beta2", "beta3", "beta4"):
            v = _to_scalar_float(getattr(self, name), name=name)
            object.__setattr__(self, name, v)

        if self.extra is not None:
            if not isinstance(self.extra, dict):
                raise TypeError("extra must be a dict {order:int -> beta_order:float} or None")
            clean: Dict[int, float] = {}
            for k, v in self.extra.items():
                if not isinstance(k, int):
                    raise TypeError(f"extra key must be int order, got {type(k)!r}")
                if k < 0:
                    raise ValueError(f"extra order must be >= 0, got {k}")
                fv = _to_scalar_float(v, name=f"extra[{k}]")
                clean[k] = fv
            object.__setattr__(self, "extra", clean)

    def get_beta_n(self, n: int) -> float:
        """Return beta_n (0..∞). Missing -> 0."""
        if not isinstance(n, int):
            raise TypeError("n must be int")
        if n < 0:
            raise ValueError("n must be >= 0")

        if self.extra is not None and n in self.extra:
            return float(self.extra[n])

        if n == 0:
            return self.beta0
        if n == 1:
            return self.beta1
        if n == 2:
            return self.beta2
        if n == 3:
            return self.beta3
        if n == 4:
            return self.beta4
        return 0.0

    def available_orders(self) -> Tuple[int, ...]:
        """
        Orders that are nonzero (based on provided params), sorted ascending.
        """
        orders = []
        for n in range(0, 5):
            if self.get_beta_n(n) != 0.0:
                orders.append(n)
        if self.extra is not None:
            for n, v in self.extra.items():
                if v != 0.0 and n not in orders:
                    orders.append(n)
        return tuple(sorted(orders))


def beta_taylor(
    omega: Union[float, np.ndarray],
    disp: DispersionParams,
    *,
    max_order: int = 4,
) -> Union[float, np.ndarray]:
    """
    Compute beta(omega) using Taylor expansion around disp.omega_ref up to max_order.

    Parameters
    ----------
    omega : float or np.ndarray
        Angular frequency omega [rad/s]
    disp : DispersionParams
        Dispersion coefficients at omega_ref
    max_order : int
        Highest order to include in expansion (>=0)

    Returns
    -------
    beta : float or np.ndarray
        Propagation constant [1/m]
    """
    if not isinstance(max_order, int):
        raise TypeError("max_order must be int")
    if max_order < 0:
        raise ValueError("max_order must be >= 0")

    w = np.asarray(omega, dtype=float)
    if not np.all(np.isfinite(w)):
        raise ValueError("omega must be finite")
    if np.any(w <= 0.0):
        raise ValueError("omega must be positive (rad/s)")

    dw = w - disp.omega_ref

    # Build series
    out = np.zeros_like(w, dtype=float)
    for n in range(0, max_order + 1):
        bn = disp.get_beta_n(n)
        if bn == 0.0:
            continue
        out = out + bn * (dw**n) / float(factorial(n))

    if np.isscalar(omega):
        return float(out.item())
    return out


def delta_beta_from_omegas(
    omegas: Sequence[float],
    disp: DispersionParams,
    *,
    max_order: int = 4,
    atol: float = 0.0,
    rtol: float = 1e-12,
) -> float:
    """
    Compute dbeta from omega1..omega4:

        dbeta = beta(omega3) + beta(omega4) - beta(omega1) - beta(omega2)

    Expects omegas in order [omega1, omega2, omega3, omega4] = [pump1, pump2, signal, idler].
    """
    om = np.asarray(list(omegas), dtype=float)
    if om.shape != (4,):
        raise ValueError(f"omegas must have shape (4,), got {om.shape}")
    if not np.all(np.isfinite(om)):
        raise ValueError("omegas must be finite")
    if np.any(om <= 0.0):
        raise ValueError("omegas must be positive (rad/s)")

    # Optional energy conservation check (exact in omega space)
    lhs = om[0] + om[1]
    rhs = om[2] + om[3]
    if not np.isclose(lhs, rhs, atol=atol, rtol=rtol):
        raise ValueError(
            "Energy conservation violated: omega1+omega2 != omega3+omega4. "
            f"(lhs={lhs:.16e}, rhs={rhs:.16e}, diff={(lhs-rhs):.16e})"
        )

    b1 = beta_taylor(om[0], disp, max_order=max_order)
    b2 = beta_taylor(om[1], disp, max_order=max_order)
    b3 = beta_taylor(om[2], disp, max_order=max_order)
    b4 = beta_taylor(om[3], disp, max_order=max_order)
    return float((b3 + b4) - (b1 + b2))


def delta_beta_symmetric(
    omega_c: float,
    omega_d: float,
    Omega: float,
    disp: DispersionParams,
    *,
    even_orders: Iterable[int] = (2, 4),
) -> float:
    """
    Compute dbeta using the symmetric definition around omega_c:

        omega1 = omega_c + omega_d
        omega2 = omega_c - omega_d
        omega3 = omega_c + omega
        omega4 = omega_c - omega

    For symmetric pairs, odd-order Taylor terms cancel exactly. Therefore:

        dbeta = Σ_{n even>=2} [ beta_n(omega_c) * (omega^n - omega_d^n) * 2 / n! ]
           = beta2(omega^2 - omega_d^2) + beta4/12 (omega^4 - omega_d^4) + beta6/360 (omega^6 - omega_d^6) + ...

    This function implements that even-order formula using coefficients from `disp`
    evaluated at disp.omega_ref, which should be set to omega_c for strict consistency.
    """
    oc = _validate_positive(omega_c, name="omega_c")
    od = _to_scalar_float(omega_d, name="omega_d")
    Om = _to_scalar_float(Omega, name="Omega")

    if disp.omega_ref != oc:
        # Not an error: you may want to approximate beta_n(omega_c) by values at a nearby omega_ref.
        # But warn early by raising only if wildly inconsistent (negative, etc.) is already checked.
        pass

    evens = list(even_orders)
    if len(evens) == 0:
        raise ValueError("even_orders must contain at least one order (e.g., 2,4)")
    for n in evens:
        if not isinstance(n, int):
            raise TypeError("even_orders must contain ints")
        if n < 2:
            raise ValueError(f"even order must be >=2, got {n}")
        if (n % 2) != 0:
            raise ValueError(f"Order must be even, got {n}")

    out = 0.0
    for n in evens:
        bn = disp.get_beta_n(n)
        if bn == 0.0:
            continue
        out += bn * (Om**n - od**n) * 2.0 / float(factorial(n))

    return float(out)


def dispersion_params_from_D_S(
    lambda_ref_m: float,
    D: float,
    S: Optional[float] = None,
    *,
    D_units: str = "SI",
    S_units: str = "SI",
    omega_ref: Optional[float] = None,
    beta0: float = 0.0,
    beta1: float = 0.0,
    beta4: float = 0.0,
    extra: Optional[Dict[int, float]] = None,
) -> DispersionParams:
    """
    Convenience builder: create DispersionParams at lambda_ref from D (and optionally S).

    Parameters
    ----------
    lambda_ref_m : float
        Reference wavelength [m] (typically around 1550e-9).
    D : float
        Dispersion parameter. Units controlled by D_units.
    S : float or None
        Dispersion slope. If provided, computes beta3. Units controlled by S_units.
    D_units : {"SI", "ps/nm/km"}
        - "SI": D is in s/m^2
        - "ps/nm/km": D is in ps/(nm*km)
    S_units : {"SI", "ps/nm^2/km"}
        - "SI": S is in s/m^3
        - "ps/nm^2/km": S is in ps/(nm^2*km)
    omega_ref : float or None
        If None, computed from lambda_ref as omega_ref = 2pic/lambda_ref.
    beta0, beta1, beta4, extra :
        Pass-through to DispersionParams.

    Returns
    -------
    DispersionParams with beta2 (and optionally beta3) populated.
    """
    lam = _validate_positive(lambda_ref_m, name="lambda_ref_m")
    if omega_ref is None:
        wref = _omega_from_lambda(lam)
    else:
        wref = _validate_positive(omega_ref, name="omega_ref")

    if D_units == "SI":
        D_SI = _to_scalar_float(D, name="D")
    elif D_units == "ps/nm/km":
        D_SI = D_ps_nm_km_to_SI(D)
    else:
        raise ValueError(f"Unknown D_units={D_units!r}. Use 'SI' or 'ps/nm/km'.")

    b2 = beta2_from_D(lam, D_SI)

    b3 = 0.0
    if S is not None:
        if S_units == "SI":
            S_SI = _to_scalar_float(S, name="S")
        elif S_units == "ps/nm^2/km":
            S_SI = S_ps_nm2_km_to_SI(S)
        else:
            raise ValueError(f"Unknown S_units={S_units!r}. Use 'SI' or 'ps/nm^2/km'.")
        b3 = beta3_from_D_S(lam, D_SI, S_SI)

    return DispersionParams(
        omega_ref=wref,
        beta0=beta0,
        beta1=beta1,
        beta2=b2,
        beta3=b3,
        beta4=beta4,
        extra=extra,
    )
