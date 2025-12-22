"""
scan_mismatch.py
scan phase mismatch
"""

from __future__ import annotations

import constants
from config import default_simulation_config, custom_simulation_config
from simulation import run_single_simulation
from plotting import plot_signal_and_idler, plot_powers

from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

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
        z_max=0.1
    )

    # --- Fixed physical parameters (your current choice) ---
    gamma = 10.0  # 1/(W*km)
    alpha = 0.0

    P1_total = 1.0  # W, total pump power (split equally)
    p_in = np.array(
        [
            P1_total / 2.0,  # pump1
            P1_total / 2.0,  # pump2
            1e-3,            # signal seed
            1e-4,            # idler seed
        ],
        dtype=float,
    )

    # Base betas and omegas
    beta0 = 5.8e9  # 1/km
    omega0 = constants.c / 1.55e-6  # rad/s
    omega = omega0 * np.ones(4, dtype=float)

    # Reference input powers (for gain definitions)
    Ps0_ref = float(p_in[2])
    Pi0_ref = float(p_in[3])

    # --- Scan range ---
    ideal_mismatch_guess = 0.0
    span = 20.0  # 1/km
    n_points = 100
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
        betas = beta0 * np.ones(4, dtype=float) + np.array([0.0, 0.0, delta, delta], dtype=float)

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
    plt.semilogy(delta_list, Gi_plot, label=f"Idler level  Gi ({gain_mode})", lw=2, ls="--")

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