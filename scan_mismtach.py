"""
scan_mismatch.py
scan phase mismatch
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

import constants
from config import default_simulation_config
from simulation import run_single_simulation
from plotting import plot_signal_and_idler, plot_powers


def scan_mismatch_seeded_signal() -> None:
    """
    Scan the beta-mismatch offset (delta added to beta_s and beta_i) and compute gain.

    Units:
        gamma: 1/(W*km)
        betas: 1/km
        z: km
    """

    cfg = default_simulation_config()

    # --- Fixed physical parameters (your current choice) ---
    gamma = 10.0  # 1/(W*km)
    alpha = 0.0

    P1_total = 1.0  # W, total pump power (split equally)
    p_in = np.array([
        P1_total / 2.0,  # pump1
        P1_total / 2.0,  # pump2
        1e-3,            # signal seed
        1e-4,             # idler seed
    ], dtype=float)

    # Base betas and omegas
    beta0 = 5.8e9  # 1/km (your code)
    omega0 = constants.c / 1.55e-6  # rad/s
    omega = omega0 * np.ones(4, dtype=float)

    # Reference signal power (for gain definitions)
    Ps0_ref = p_in[2]
    Ps1_ref = p_in[3]

    # --- Scan range ---
    # Previous "single-pump-inspired" guess:
    ideal_mismatch_guess = -(2.0 / 3.0) * gamma * P1_total  # 1/km

    span = 30.0  # 1/km
    n_points = 200

    delta_list = np.linspace(ideal_mismatch_guess - span, ideal_mismatch_guess + span, n_points)

    # Storage
    Gs = np.empty_like(delta_list)
    Gi = np.empty_like(delta_list)

    # Optional: store also final signal/idler powers
    PsL = np.empty_like(delta_list)
    PiL = np.empty_like(delta_list)

    # --- Run scan ---
    for k, delta in enumerate(delta_list):
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

        Ps0 = Ps[0]
        Ps_end = Ps[-1]
        Pi_end = Pi[-1]

        # Gain definitions
        # - Signal gain relative to its own input value
        # - Idler "gain" relative to seeded signal reference to avoid division by ~0
        eps = 1e-30
        Gs[k] = Ps_end / (Ps0 + eps)
        Gi[k] = Pi_end / (Ps1_ref + eps)

        PsL[k] = Ps_end
        PiL[k] = Pi_end

    # --- Find best mismatch for signal gain ---
    best_idx = int(np.argmax(Gs))
    best_delta = float(delta_list[best_idx])
    best_Gs = float(Gs[best_idx])
    best_Gi = float(Gi[best_idx])

    print("=== Mismatch scan results ===")
    print(f"gamma = {gamma:.6g} 1/(W*km)")
    print(f"Total pump power P1_total = {P1_total:.6g} W  (split: {P1_total/2:.6g} + {P1_total/2:.6g} W)")
    print(f"Seed signal Ps(0) ~ {Ps0_ref:.6g} W")
    print(f"Ideal_mismatch_guess (your formula) = {ideal_mismatch_guess:.6g} 1/km")
    print("--- Best point (max signal gain) ---")
    print(f"best_delta = {best_delta:.6g} 1/km")
    print(f"Signal gain Gs = Ps(L)/Ps(0) = {best_Gs:.6g}")
    print(f"Idler relative level Gi = Pi(L)/Ps_seed(0) = {best_Gi:.6g}")

    plt.figure(figsize=(8, 5))

    # Protect against zeros or negative values
    Gs_plot = np.clip(Gs, 1e-20, None)
    Gi_plot = np.clip(Gi, 1e-20, None)

    plt.semilogy(delta_list, Gs_plot, label="Signal gain  Gs", lw=2)
    plt.semilogy(delta_list, Gi_plot, label="Idler level  Gi", lw=2, ls="--")

    plt.axvline(best_delta, color="k", ls=":", lw=1.5,
                label=f"best delta = {best_delta:.3g} 1/km")

    plt.xlabel(r"$\delta$  [1/km]")
    plt.ylabel("Gain (log scale)")
    plt.title("Parametric gain vs phase-mismatch (log scale)")

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

    plot_signal_and_idler(z_best, A_best, title=f"Best delta = {best_delta:.3g} 1/km", z_unit="km")
    plot_powers(z_best, A_best, title=f"Powers at best delta = {best_delta:.3g} 1/km", z_unit="km")

    # --- Print a small table around optimum ---
    lo = max(0, best_idx - 3)
    hi = min(len(delta_list), best_idx + 4)

    print("--- Local neighborhood ---")
    for j in range(lo, hi):
        print(f"delta={delta_list[j]: .6g}  Gs={Gs[j]: .6g}  PsL={PsL[j]: .6g}  PiL={PiL[j]: .6g}")
