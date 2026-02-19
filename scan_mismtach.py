"""
scan_mismatch.py
scan phase mismatch
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Literal
import numpy as np
import matplotlib.pyplot as plt

import constants
from config import SimulationConfig
from dispersion import DispersionParams
from frequency_plan import plan_from_wavelengths
from phase_matching import PhaseMatchingConfig, PhaseMatchingMethod, compute_phase_mismatch
from simulation import run_single_simulation, custom_simulation_config
from plotting import plot_powers, plot_signal_and_idler


import time
from tqdm import tqdm  # pip install tqdm

GainMode = Literal["end", "max"]


def _select_power_metric(Pz: np.ndarray, mode: GainMode) -> float:
    """
    Select which power value to use as the "output" metric.

    mode="end": use P(z_max) = Pz[-1]
    mode="max": use max_z P(z) = Pz.max()
    """
    if Pz.ndim != 1:
        raise ValueError("Pz must be a 1D array of power versus z.")
    if mode == "end":
        return float(Pz[-1])
    if mode == "max":
        return float(np.max(Pz))
    raise ValueError(f"Unknown gain_mode={mode!r}. Use 'end' or 'max'.")


def scan_mismatch_seeded_signal(gain_mode: "GainMode" = "end") -> None:
    """
    Scan the beta-mismatch offset (delta added to beta_s and beta_i) and compute gain.

    gain_mode:
        "end" -> gain computed as P(z_max)/P(0)
        "max" -> gain computed as max_z P(z)/P(0)

    Units:
        gamma: 1/(W*km)
        betas: 1/km
        z: km
    """
    cfg = custom_simulation_config(
        z_max=0.5,
        dz=1e-3
    )

    # --- Fixed physical parameters (your current choice) ---
    gamma = 10.0  # 1/(W*km)
    alpha = 0.0

    P1_total = 0.1  # W, total pump power (split equally)
    p_in = np.array(
        [
            P1_total,  # pump1
            P1_total,  # pump2
            1e-5,            # signal seed
            0,            # idler seed
        ],
        dtype=float,
    )

    # Base betas and omegas
    beta0 = 5.8e9  # 1/km
    omega0 = constants.c / 1.55e-6  # rad/s
    omega = omega0 * np.ones(4, dtype=float)

    # Reference input powers (for gain definitions)
    Ps0_ref = float(p_in[2])
    Pi0_ref = float(p_in[2])

    # --- Scan range ---
    ideal_mismatch_guess = 0.0
    span = 40.0  # 1/km
    n_points = 200
    delta_list = np.linspace(
        ideal_mismatch_guess - span,
        ideal_mismatch_guess + span,
        n_points,
    )

    # Storage
    Gs = np.empty_like(delta_list)
    Gi = np.empty_like(delta_list)

    # Store selected metric values (either end or max) for debugging/printing
    Ps_metric_arr = np.empty_like(delta_list)
    Pi_metric_arr = np.empty_like(delta_list)

    metric_label = "P(z_max)" if gain_mode == "end" else "max_z P(z)"

    print("=== Starting mismatch scan ===")
    print(f"n_points = {n_points}")
    print(f"delta range = [{float(delta_list[0]):.6g}, {float(delta_list[-1]):.6g}] 1/km")
    print(f"Gain metric mode = {gain_mode!r}  -> using {metric_label}")

    # --- Run scan with timer + progress bar ---
    eps = 1e-30
    t0 = time.perf_counter()

    best_idx_running = 0
    best_gs_running = -np.inf

    bar = tqdm(
        enumerate(delta_list),
        total=len(delta_list),
        desc="Scanning mismatch",
        unit="pt",
        dynamic_ncols=True,
        leave=True,
    )

    for k, delta in bar:
        betas = beta0 * np.ones(4, dtype=float) + np.array([0.0, 0.0, 0.0, delta], dtype=float)

        z, A = run_single_simulation(
            cfg,
            gamma=gamma,
            alpha=alpha,
            beta=betas,
            omega=omega,
            p_in=p_in,
            phase_in=None,
        )

        P = np.abs(A) ** 2
        Ps = P[:, 2]
        Pi = P[:, 3]

        Ps0 = float(Ps[0])

        Ps_metric = _select_power_metric(Ps, gain_mode)
        Pi_metric = _select_power_metric(Pi, gain_mode)

        # Gain definitions
        gs_val = Ps_metric / (Ps0 + eps)
        gi_val = Pi_metric / (Pi0_ref + eps)

        Gs[k] = gs_val
        Gi[k] = gi_val

        Ps_metric_arr[k] = Ps_metric
        Pi_metric_arr[k] = Pi_metric

        if gs_val > best_gs_running:
            best_gs_running = float(gs_val)
            best_idx_running = int(k)

        elapsed = time.perf_counter() - t0
        avg = elapsed / (k + 1)
        bar.set_postfix(
            delta=f"{float(delta):.3g}",
            Gs=f"{float(gs_val):.3g}",
            Gi=f"{float(gi_val):.3g}",
            bestGs=f"{best_gs_running:.3g}",
            avg_s=f"{avg:.3f}",
        )

    t1 = time.perf_counter()
    elapsed_total = t1 - t0
    avg_per_point = elapsed_total / max(1, len(delta_list))
    pts_per_sec = (len(delta_list) / elapsed_total) if elapsed_total > 0 else float("inf")

    print("=== Timing ===")
    print(f"Elapsed total: {elapsed_total:.3f} s")
    print(f"Avg per point: {avg_per_point:.4f} s/pt")
    print(f"Throughput:    {pts_per_sec:.2f} pt/s")

    # --- Find best mismatch for signal gain ---
    best_idx = int(np.argmax(Gs))
    best_delta = float(delta_list[best_idx])
    best_Gs = float(Gs[best_idx])
    best_Gi = float(Gi[best_idx])

    print("=== Mismatch scan results ===")
    print(f"gamma = {gamma:.6g} 1/(W*km)")
    print(f"Total pump power P1_total = {P1_total:.6g} W  (split: {P1_total/2:.6g} + {P1_total/2:.6g} W)")
    print(f"Seed signal Ps(0) = {Ps0_ref:.6g} W")
    print(f"Seed idler  Pi(0) = {Pi0_ref:.6g} W")
    print(f"Ideal_mismatch_guess = {ideal_mismatch_guess:.6g} 1/km")
    print(f"Gain metric mode = {gain_mode!r}  -> using {metric_label}")
    print("--- Best point (max signal gain) ---")
    print(f"best_delta = {best_delta:.6g} 1/km")
    print(f"Signal gain Gs = {metric_label}/Ps(0) = {best_Gs:.6g}")
    print(f"Idler  level Gi = {metric_label}/Pi(0) = {best_Gi:.6g}")
    print(f"(Progress bar best-so-far index during scan was {best_idx_running}, final best index is {best_idx}.)")

    # --- Plot gain vs mismatch ---
    plt.figure(figsize=(8, 5))

    Gs_plot = np.clip(Gs, 1e-20, None)
    Gi_plot = np.clip(Gi, 1e-20, None)

    plt.semilogy(delta_list, Gs_plot, label=f"Signal gain  Gs ({gain_mode})", lw=2)
    # plt.semilogy(delta_list, Gi_plot, label=f"Idler level  Gi ({gain_mode})", lw=2, ls="--")

    plt.axvline(best_delta, color="k", ls=":", lw=1.5, label=f"best delta = {best_delta:.3g} 1/km")

    plt.xlabel(r"$\delta$  [1/km]")
    plt.ylabel("Gain (log scale)")
    plt.title(f"Parametric gain vs phase-mismatch (metric: {metric_label})")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Re-run the best case and plot ---
    betas_best = beta0 * np.ones(4, dtype=float) + np.array([0.0, 0.0, best_delta, best_delta], dtype=float)

    z_best, A_best = run_single_simulation(
        cfg,
        gamma=gamma,
        alpha=alpha,
        beta=betas_best,
        omega=omega,
        p_in=p_in,
        phase_in=None,
    )

    # Optional: if gain_mode="max", print where the peak occurs
    if gain_mode == "max":
        P_best = np.abs(A_best) ** 2
        Ps_best = P_best[:, 2]
        Pi_best = P_best[:, 3]
        iz_ps = int(np.argmax(Ps_best))
        iz_pi = int(np.argmax(Pi_best))
        print("--- Peak locations for best delta ---")
        print(f"Signal peak at z = {float(z_best[iz_ps]):.6g} km, Ps_max = {float(Ps_best[iz_ps]):.6g} W")
        print(f"Idler  peak at z = {float(z_best[iz_pi]):.6g} km, Pi_max = {float(Pi_best[iz_pi]):.6g} W")

    plot_signal_and_idler(z_best, A_best, title=f"Best delta = {best_delta:.3g} 1/km", z_unit="km")
    plot_powers(z_best, A_best, title=f"Powers at best delta = {best_delta:.3g} 1/km", z_unit="km")

    # --- Print a small table around optimum ---
    lo = max(0, best_idx - 3)
    hi = min(len(delta_list), best_idx + 4)

    print("--- Local neighborhood ---")
    for j in range(lo, hi):
        print(
            f"delta={delta_list[j]: .6g}  "
            f"Gs={Gs[j]: .6g}  "
            f"Ps_metric={Ps_metric_arr[j]: .6g}  "
            f"Pi_metric={Pi_metric_arr[j]: .6g}"
        )


def plot_max_signal_gain_vs_lambda_signal(
    *,
    cfg: SimulationConfig,
    lambda_p1_m: float,
    lambda_p2_m: float,
    lambda_signal_m: Sequence[float],
    gamma: float,
    alpha: float,
    p_in: Sequence[float],
    phase_in: Optional[Sequence[float]] = None,
    dispersion: Optional[DispersionParams] = None,
    phase_matching_cfg: Optional[PhaseMatchingConfig] = None,
    length_unit: str = "m",
    return_wavelength_unit: str = "nm",
    gain_unit: str = "dB",              # "dB" or "linear"
    xscale: str = "linear",             # "linear" or "log"
    yscale: str = "linear",             # "linear" or "log" (log only makes sense for gain_unit="linear")
    show_progress: bool = True,         # tqdm progress bar
    tqdm_desc: str = "Sweeping λ3",     # progress bar label
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scan the signal wavelength λ3 and plot the *maximum* signal gain versus λ3.

    Wave order is fixed:
        [pump1, pump2, signal, idler] -> [1, 2, 3, 4]

    For each λ3 in lambda_signal_m:
        - build ω = [ω1, ω2, ω3, ω4] from (λ1, λ2, λ3) (idler inferred by ω4 = ω1+ω2-ω3)
        - run a single simulation
        - compute instantaneous signal power P3(z) = |A3(z)|^2
        - compute gain(z) = P3(z) / P3(0) and take max over z
        - report either:
            * gain_unit="linear": G_max = max_z gain(z)
            * gain_unit="dB":     G_max_dB = 10 log10(G_max)

    Log scales:
        - xscale can be "linear" or "log"
        - yscale can be "linear" or "log"
          NOTE: yscale="log" is only supported for gain_unit="linear" (since dB can be <= 0).

    Returns
    -------
    x_wavelength : np.ndarray
        Wavelength array in units specified by return_wavelength_unit.
    gain_max : np.ndarray
        Maximum signal gain for each wavelength in chosen gain_unit.
        If a run fails, the corresponding entry is NaN.
    """
    lam1 = float(lambda_p1_m)
    lam2 = float(lambda_p2_m)

    lam3_arr = np.asarray(list(lambda_signal_m), dtype=float)
    if lam3_arr.ndim != 1 or lam3_arr.size == 0:
        raise ValueError("lambda_signal_m must be a non-empty 1D sequence")
    if not np.all(np.isfinite(lam3_arr)) or np.any(lam3_arr <= 0.0):
        raise ValueError("lambda_signal_m must contain finite positive wavelengths (m)")

    p0 = np.asarray(list(p_in), dtype=float)
    if p0.shape != (4,):
        raise ValueError(f"p_in must have shape (4,), got {p0.shape}")
    if not np.all(np.isfinite(p0)) or np.any(p0 < 0.0):
        raise ValueError("p_in must contain finite non-negative powers")
    if p0[2] <= 0.0:
        raise ValueError("p_in[2] (signal seed power) must be > 0 to define gain")

    ph0 = None
    if phase_in is not None:
        ph0 = np.asarray(list(phase_in), dtype=float)
        if ph0.shape != (4,):
            raise ValueError(f"phase_in must have shape (4,), got {ph0.shape}")
        if not np.all(np.isfinite(ph0)):
            raise ValueError("phase_in must contain finite values")

    gain_unit_norm = str(gain_unit).strip().lower()
    if gain_unit_norm not in ("db", "linear"):
        raise ValueError("gain_unit must be 'dB' or 'linear'")

    xscale_norm = str(xscale).strip().lower()
    yscale_norm = str(yscale).strip().lower()
    if xscale_norm not in ("linear", "log"):
        raise ValueError("xscale must be 'linear' or 'log'")
    if yscale_norm not in ("linear", "log"):
        raise ValueError("yscale must be 'linear' or 'log'")

    if yscale_norm == "log" and gain_unit_norm == "db":
        raise ValueError("yscale='log' is not supported with gain_unit='dB'. Use gain_unit='linear'.")

    gain_max = np.full(lam3_arr.shape, np.nan, dtype=float)

    iterator = range(lam3_arr.size)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=tqdm_desc, total=lam3_arr.size)

    for i in iterator:
        lam3 = float(lam3_arr[i])
        try:
            omega = plan_from_wavelengths(lam1, lam2, lam3, lambda4_m=None)

            z, A = run_single_simulation(
                cfg,
                gamma=gamma,
                alpha=alpha,
                omega=omega,
                p_in=p0,
                phase_in=ph0,
                dispersion=dispersion,
                phase_matching_cfg=phase_matching_cfg,
                beta_legacy=None,
                length_unit=length_unit,
                return_length_unit=length_unit,
            )

            P3_z = np.abs(A[:, 2]) ** 2
            if not np.all(np.isfinite(P3_z)):
                gain_max[i] = np.nan
                continue

            g_lin = float(np.max(P3_z) / p0[2])
            if not np.isfinite(g_lin) or g_lin <= 0.0:
                gain_max[i] = np.nan
                continue

            if gain_unit_norm == "linear":
                gain_max[i] = g_lin
            else:
                gain_max[i] = 10.0 * np.log10(g_lin)

        except Exception:
            gain_max[i] = np.nan

    # x-axis units
    if return_wavelength_unit.strip().lower() == "nm":
        x = lam3_arr * 1e9
        x_label = r"Signal wavelength $\lambda_3$ (nm)"
    elif return_wavelength_unit.strip().lower() == "m":
        x = lam3_arr
        x_label = r"Signal wavelength $\lambda_3$ (m)"
    else:
        raise ValueError("return_wavelength_unit must be 'm' or 'nm'")

    # y-axis label
    if gain_unit_norm == "linear":
        y = gain_max
        y_label = r"Max signal gain $G_{\max}$ (linear)"
    else:
        y = gain_max
        y_label = r"Max signal gain $G_{\max}$ (dB)"

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(r"Maximum signal gain vs signal wavelength")
    plt.grid(True, which="both")

    plt.xscale(xscale_norm)
    plt.yscale(yscale_norm)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return x, gain_max


def _omega0_from_dispersion(disp: DispersionParams) -> float:
    # Keep this tolerant to small naming variations.
    for name in ("omega0", "omega0_rad_s", "w0", "w0_rad_s"):
        if hasattr(disp, name):
            return float(getattr(disp, name))
    raise AttributeError("DispersionParams must define omega0 in rad/s (e.g., field 'omega0').")


def _beta_taylor(disp: DispersionParams, omega: float) -> float:
    """
    β(ω) ≈ β0 + β1 d + (β2/2) d^2 + (β3/6) d^3 + (β4/24) d^4,  d = ω-ω0
    Missing coefficients are treated as 0.
    """
    w0 = _omega0_from_dispersion(disp)
    d = float(omega) - w0

    beta0 = float(getattr(disp, "beta0", 0.0))
    beta1 = float(getattr(disp, "beta1", 0.0))
    beta2 = float(getattr(disp, "beta2", 0.0))
    beta3 = float(getattr(disp, "beta3", 0.0))
    beta4 = float(getattr(disp, "beta4", 0.0))

    d2 = d * d
    d3 = d2 * d
    d4 = d2 * d2

    return beta0 + beta1 * d + 0.5 * beta2 * d2 + (1.0 / 6.0) * beta3 * d3 + (1.0 / 24.0) * beta4 * d4


def _delta_beta_from_omegas(disp: DispersionParams, omega: np.ndarray) -> float:
    """
    dBeta = β(ω1)+β(ω2)-β(ω3)-β(ω4), wave order [1,2,3,4] = [p1,p2,s,i].
    Units: 1/length-unit used by βk coefficients (typically 1/m or 1/km).
    """
    w = np.asarray(omega, dtype=float)
    if w.shape != (4,):
        raise ValueError("omega must have shape (4,) = [ω1, ω2, ω3, ω4]")
    return (_beta_taylor(disp, w[0]) + _beta_taylor(disp, w[1]) - _beta_taylor(disp, w[2]) - _beta_taylor(disp, w[3]))


def plot_dbeta_vs_lambda_signal(
    *,
    gamma: float,
    lambda_p1_m: float,
    lambda_p2_m: float,
    lambda_signal_m: Sequence[float],
    p_in: Sequence[float],
    dispersion: DispersionParams,
    return_wavelength_unit: str = "nm",   # "nm" | "m"
    xscale: str = "linear",               # "linear" | "log"
    yscale: str = "linear",               # "linear" | "log"
    length_unit: str = "m",               # affects only axis label: "m" | "km"
    show_progress: bool = True,
    tqdm_desc: str = r"Scanning dBeta(λ3)",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Plot dBeta(λ3) and overlay a dashed horizontal line at gamma*(P1+P2).

    Returns
    -------
    x_wavelength : np.ndarray
        λ3 array in chosen return_wavelength_unit.
    dbeta : np.ndarray
        dBeta(λ3) in 1/length_unit (consistent with your βk coefficients).
    """
    lam1 = float(lambda_p1_m)
    lam2 = float(lambda_p2_m)

    lam3_arr = np.asarray(list(lambda_signal_m), dtype=float)
    if lam3_arr.ndim != 1 or lam3_arr.size == 0:
        raise ValueError("lambda_signal_m must be a non-empty 1D sequence")
    if not np.all(np.isfinite(lam3_arr)) or np.any(lam3_arr <= 0.0):
        raise ValueError("lambda_signal_m must contain finite positive wavelengths (m)")

    p0 = np.asarray(list(p_in), dtype=float)
    if p0.shape != (4,):
        raise ValueError(f"p_in must have shape (4,), got {p0.shape}")
    if not np.all(np.isfinite(p0)) or np.any(p0 < 0.0):
        raise ValueError("p_in must contain finite non-negative powers")

    xscale_norm = str(xscale).strip().lower()
    yscale_norm = str(yscale).strip().lower()
    if xscale_norm not in ("linear", "log"):
        raise ValueError("xscale must be 'linear' or 'log'")
    if yscale_norm not in ("linear", "log"):
        raise ValueError("yscale must be 'linear' or 'log'")

    dbeta = np.full(lam3_arr.shape, np.nan, dtype=float)

    iterator = range(lam3_arr.size)
    if show_progress and tqdm is not None:
        iterator = tqdm(iterator, desc=tqdm_desc, total=lam3_arr.size)

    for i in iterator:
        lam3 = float(lam3_arr[i])
        try:
            omega = plan_from_wavelengths(lam1, lam2, lam3, lambda4_m=None)
            dbeta[i] = _delta_beta_from_omegas(dispersion, omega)
        except Exception:
            dbeta[i] = np.nan

    # x-axis units
    unit = return_wavelength_unit.strip().lower()
    if unit == "nm":
        x = lam3_arr * 1e9
        x_label = r"Signal wavelength $\lambda_3$ (nm)"
    elif unit == "m":
        x = lam3_arr
        x_label = r"Signal wavelength $\lambda_3$ (m)"
    else:
        raise ValueError("return_wavelength_unit must be 'm' or 'nm'")

    y_unit = "1/km" if str(length_unit).strip().lower() == "km" else "1/m"

    # reference line: gamma*(P1+P2)
    ref = float(gamma) * float(p0[0] + p0[1])

    # log y-scale safety
    if yscale_norm == "log":
        if not np.all(np.isfinite(dbeta)):
            pass
        if np.nanmin(dbeta) <= 0.0 or ref <= 0.0:
            raise ValueError("yscale='log' requires dBeta and gamma*(P1+P2) to be strictly > 0.")

    plt.figure(figsize=(8.0, 5.0))
    plt.plot(x, dbeta, label=r"$d\beta(\lambda_3)$")
    plt.axhline(ref, linestyle="--", label=r"$\gamma(P_1+P_2)$")

    plt.xlabel(x_label)
    plt.ylabel(rf"$d\beta$ [{y_unit}]")

    plt.xscale(xscale_norm)
    plt.yscale(yscale_norm)

    if title is not None:
        plt.title(title)

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    return x, dbeta


def plot_max_gain_and_dbeta_vs_lambda_signal(
    *,
    cfg: SimulationConfig,
    lambda_p1_m: float,
    lambda_p2_m: float,
    lambda_signal_m: Sequence[float],
    gamma: float,
    alpha: float,
    p_in: Sequence[float],
    phase_in: Optional[Sequence[float]] = None,
    dispersion: DispersionParams,
    phase_matching_cfg: Optional[PhaseMatchingConfig] = None,
    length_unit: str = "m",
    return_wavelength_unit: str = "nm",
    gain_unit: str = "dB",              # "dB" or "linear"
    xscale: str = "linear",             # "linear" or "log"
    yscale_gain: str = "linear",        # "linear" or "log" (log only for gain_unit="linear")
    yscale_dbeta: str = "linear",       # "linear" or "log" (log requires strictly >0 values)
    show_progress: bool = True,
    tqdm_desc: str = "Sweeping λ3 (gain + dBeta)",
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single sweep over λ3 that computes BOTH:
      1) max signal gain vs λ3 (same metric as plot_max_signal_gain_vs_lambda_signal)
      2) dBeta(λ3) from phase-matching layer,
         and overlays a dashed horizontal line at gamma*(P1+P2).

    Returns
    -------
    x_wavelength : np.ndarray
        λ3 array in units specified by return_wavelength_unit.
    gain_max : np.ndarray
        Max signal gain for each λ3 in chosen gain_unit.
    dbeta : np.ndarray
        Phase mismatch Δβ for each λ3 in units 1/length_unit (consistent with gamma).
    """
    # --- Validate inputs ---
    lam1 = float(lambda_p1_m)
    lam2 = float(lambda_p2_m)

    lam3_arr = np.asarray(list(lambda_signal_m), dtype=float)
    if lam3_arr.ndim != 1 or lam3_arr.size == 0:
        raise ValueError("lambda_signal_m must be a non-empty 1D sequence")
    if not np.all(np.isfinite(lam3_arr)) or np.any(lam3_arr <= 0.0):
        raise ValueError("lambda_signal_m must contain finite positive wavelengths (m)")

    p0 = np.asarray(list(p_in), dtype=float)
    if p0.shape != (4,):
        raise ValueError(f"p_in must have shape (4,), got {p0.shape}")
    if not np.all(np.isfinite(p0)) or np.any(p0 < 0.0):
        raise ValueError("p_in must contain finite non-negative powers")
    if p0[2] <= 0.0:
        raise ValueError("p_in[2] (signal seed power) must be > 0 to define gain")

    ph0 = None
    if phase_in is not None:
        ph0 = np.asarray(list(phase_in), dtype=float)
        if ph0.shape != (4,):
            raise ValueError(f"phase_in must have shape (4,), got {ph0.shape}")
        if not np.all(np.isfinite(ph0)):
            raise ValueError("phase_in must contain finite values")

    if dispersion is None:
        raise ValueError("dispersion must be provided to compute dBeta(λ3)")

    gain_unit_norm = str(gain_unit).strip().lower()
    if gain_unit_norm not in ("db", "linear"):
        raise ValueError("gain_unit must be 'dB' or 'linear'")

    xscale_norm = str(xscale).strip().lower()
    yscale_gain_norm = str(yscale_gain).strip().lower()
    yscale_dbeta_norm = str(yscale_dbeta).strip().lower()

    if xscale_norm not in ("linear", "log"):
        raise ValueError("xscale must be 'linear' or 'log'")
    if yscale_gain_norm not in ("linear", "log"):
        raise ValueError("yscale_gain must be 'linear' or 'log'")
    if yscale_dbeta_norm not in ("linear", "log"):
        raise ValueError("yscale_dbeta must be 'linear' or 'log'")

    if yscale_gain_norm == "log" and gain_unit_norm == "db":
        raise ValueError("yscale_gain='log' is not supported with gain_unit='dB'. Use gain_unit='linear'.")

    # Default PM config matches your runner default when dispersion is present
    pm_cfg = phase_matching_cfg
    if pm_cfg is None:
        pm_cfg = PhaseMatchingConfig(
            method=PhaseMatchingMethod.SYMMETRIC_EVEN,
            max_order=4,
            even_orders=(2, 4),
            atol=0.0,
            rtol=1e-12,
            provided_delta_beta=None,
        )

    # --- Allocate outputs ---
    gain_max = np.full(lam3_arr.shape, np.nan, dtype=float)
    dbeta = np.full(lam3_arr.shape, np.nan, dtype=float)

    iterator = range(lam3_arr.size)
    if show_progress:
        iterator = tqdm(iterator, desc=tqdm_desc, total=lam3_arr.size)

    # --- Single sweep loop ---
    for i in iterator:
        lam3 = float(lam3_arr[i])
        try:
            omega = plan_from_wavelengths(lam1, lam2, lam3, lambda4_m=None)

            # dBeta in units consistent with `dispersion` and `phase_matching_cfg`
            pm_res = compute_phase_mismatch(
                omegas=omega,
                disp=dispersion,
                cfg=pm_cfg,
                symmetric_hint=None,
            )
            dbeta[i] = float(pm_res.delta_beta)

            # Run simulation (internally converts units as needed)
            z, A = run_single_simulation(
                cfg,
                gamma=gamma,
                alpha=alpha,
                omega=omega,
                p_in=p0,
                phase_in=ph0,
                dispersion=dispersion,
                phase_matching_cfg=pm_cfg,
                beta_legacy=None,
                length_unit=length_unit,
                return_length_unit=length_unit,
            )

            P3_z = np.abs(A[:, 2]) ** 2
            if not np.all(np.isfinite(P3_z)):
                continue

            g_lin = float(np.max(P3_z) / p0[2])
            if not np.isfinite(g_lin) or g_lin <= 0.0:
                continue

            if gain_unit_norm == "linear":
                gain_max[i] = g_lin
            else:
                gain_max[i] = 10.0 * np.log10(g_lin)

        except Exception:
            # leave NaNs
            continue

    # --- x-axis units ---
    if return_wavelength_unit.strip().lower() == "nm":
        x = lam3_arr * 1e9
        x_label = r"Signal wavelength $\lambda_3$ (nm)"
    elif return_wavelength_unit.strip().lower() == "m":
        x = lam3_arr
        x_label = r"Signal wavelength $\lambda_3$ (m)"
    else:
        raise ValueError("return_wavelength_unit must be 'm' or 'nm'")

    # --- Reference line gamma*(P1+P2) (units: 1/length_unit) ---
    ref_line = -float(gamma) * float(p0[0] + p0[1])

    # --- Plot (two stacked subplots, shared x) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    # Top: gain
    ax1.plot(x, gain_max, marker="o")
    ax1.set_ylabel("Max signal gain (linear)" if gain_unit_norm == "linear" else "Max signal gain (dB)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_yscale(yscale_gain_norm)

    # Bottom: dBeta + reference line
    ax2.plot(x, dbeta, marker="o", label=r"$\Delta\beta(\lambda_3)$")
    ax2.axhline(ref_line, ls="--", lw=2, label=r"$\gamma(P_1+P_2)$")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(rf"$\Delta\beta$  [1/{length_unit}]")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xscale(xscale_norm)
    ax2.set_yscale(yscale_dbeta_norm)
    ax2.legend()

    fig.suptitle("Max signal gain and phase mismatch vs signal wavelength")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return x, gain_max, dbeta
