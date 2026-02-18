# main.py
from __future__ import annotations

import numpy as np

import plotting
from config import custom_simulation_config
from dispersion import dispersion_params_from_D_S, delta_beta_from_omegas, delta_beta_symmetric
from frequency_plan import (
    plan_from_wavelengths,
    infer_symmetry_from_omegas,
    lambda_from_omega,
    describe_plan,
)
from phase_matching import PhaseMatchingConfig, PhaseMatchingMethod
from simulation import run_single_simulation
from scan_mismtach import plot_max_signal_gain_vs_lambda_signal


def main_single_simulation() -> None:
    # ----------------------------
    # 1) Numerical grid (meters)
    # ----------------------------
    # Fiber length 500 m, step 0.1 m
    cfg = custom_simulation_config(z_max=500.0, dz=0.1)

    # ----------------------------
    # 2) Frequency plan (dual-pump)
    #    Order: [pump1, pump2, signal, idler]
    # ----------------------------
    lambda1 = 1525e-9  # pump1 (m)
    lambda2 = 1575e-9  # pump2 (m)
    lambda3 = 1530e-9  # signal (m)
    omega = plan_from_wavelengths(lambda1, lambda2, lambda3, lambda4_m=None)

    # (Optional) print the plan for sanity
    print(describe_plan(omega))

    # Infer symmetric variables to define ωc (useful as dispersion expansion point)
    sp = infer_symmetry_from_omegas(omega1=omega[0], omega2=omega[1], omega3=omega[2], omega4=omega[3])
    lambda_c = lambda_from_omega(sp.omega_c)

    # ----------------------------
    # 3) Dispersion (at ωc / λc)
    #    Example values at ~1550 nm:
    #       D = 5 ps/(nm·km), S = 0.025 ps/(nm^2·km)
    #    Note: beta4 is not provided by D,S -> set beta4=0 for now.
    # ----------------------------
    disp = dispersion_params_from_D_S(
        lambda_ref_m=lambda_c,
        D=5.0,
        S=0.025,
        D_units="ps/nm/km",
        S_units="ps/nm^2/km",
        omega_ref=sp.omega_c
    )

    # Use the symmetric even-order mismatch formula: Δβ ≈ β2(Ω^2-ωd^2) + β4/12(Ω^4-ωd^4)
    pm_cfg = PhaseMatchingConfig(
        method=PhaseMatchingMethod.SYMMETRIC_EVEN,
        even_orders=(2, 4),
        max_order=4,
        atol=0.0,
        rtol=1e-12,
        provided_delta_beta=None,
    )

    # ----------------------------
    # 4) Nonlinearity + loss (per meter)
    # ----------------------------
    gamma_km = 10.0  # 1/(W·km)
    gamma_m = gamma_km / 1000.0  # 1/(W·m)

    alpha_db_per_km = 0.01  # dB/km (power loss)
    alpha_m = (np.log(10.0) / 10.0) * alpha_db_per_km / 1000.0  # 1/m

    # ----------------------------
    # 5) Inputs
    # ----------------------------
    p_in = np.array([0.3, 0.3, 1e-3, 1e-4], dtype=float)  # W
    phase_in = np.zeros(4, dtype=float)  # rad

    # ----------------------------
    # 6) Run
    # ----------------------------
    z, A = run_single_simulation(
        cfg,
        gamma=gamma_m,
        alpha=alpha_m,
        omega=omega,
        p_in=p_in,
        phase_in=phase_in,
        dispersion=disp,
        phase_matching_cfg=pm_cfg,
        beta_legacy=None,
        length_unit="m",
        return_length_unit="m",
    )

    # ----------------------------
    # 7) Report results
    # ----------------------------
    Pz = np.abs(A) ** 2
    P_out = Pz[-1]
    gain_signal_db = 10.0 * np.log10(P_out[2] / p_in[2])
    db = delta_beta_from_omegas(omegas=omega, disp=disp)

    db1 = delta_beta_symmetric(omega_c=sp.omega_c, omega_d=sp.omega_d, Omega=sp.Omega, disp=disp)

    print("\n--- Results ---")
    print(f"z_end = {z[-1]:.3f} m")
    print(f"P_in  [W] = {p_in}")
    print(f"P_out [W] = {P_out}")
    print(f"Signal gain = {gain_signal_db:.3f} dB")
    print(f"dbeta = {db:.3f} m^-1")
    print(f"dbeta_sym = {db1:.3f} m^-1")
    print(f"gamma(P1 + P2) = {gamma_m * (p_in[0] + p_in[1]):.3f} m^-1")

    plotting.plot_fwm_sbs_powers_forward(z, A, scale="dbW")

def main_gain_spectrum():
    # ----------------------------
    # 1) Pumps and signal scan (meters)
    # ----------------------------
    lambda_p1 = 1540e-9  # pump1
    lambda_p2 = 1550e-9  # pump2

    # Signal scan: 1520..1580 nm
    lambda_signal = np.linspace(1520e-9, 1580e-9, 61)  # 1 nm step

    # ----------------------------
    # 2) Simulation grid (meters)
    # ----------------------------
    # Keep it moderate; scanning 61 wavelengths can be expensive if dz is tiny.
    # 200 m fiber with dz=0.2 m -> 1000 steps per run.
    cfg = custom_simulation_config(z_max=200.0, dz=0.2)

    # ----------------------------
    # 3) Dispersion reference at ωc (depends only on pumps)
    # ----------------------------
    # Use ω from any signal point (ωc depends only on pumps, not on λ3)
    omega_ref = plan_from_wavelengths(lambda_p1, lambda_p2, float(lambda_signal[0]), lambda4_m=None)
    sp = infer_symmetry_from_omegas(
        omega1=omega_ref[0], omega2=omega_ref[1], omega3=omega_ref[2], omega4=omega_ref[3]
    )
    lambda_c = lambda_from_omega(sp.omega_c)

    # Example dispersion parameters near 1550 nm (replace with your fiber data if needed)
    disp = dispersion_params_from_D_S(
        lambda_ref_m=lambda_c,
        D=1,
        S=0.005,
        dSdlmbd=0,
        D_units="ps/nm/km",
        S_units="ps/nm^2/km",
        dSdlmbd_units="ps/nm^3/km",
        omega_ref=sp.omega_c
    )

    # Phase mismatch method consistent with your dispersion-sheet form
    pm_cfg = PhaseMatchingConfig(
        method=PhaseMatchingMethod.SYMMETRIC_EVEN,
        even_orders=(2, 4),
        max_order=4,
        atol=0.0,
        rtol=1e-12,
        provided_delta_beta=None,
    )

    # ----------------------------
    # 4) Fiber nonlinearity and loss (per meter)
    # ----------------------------
    gamma_km = 10.0  # 1/(W·km)
    gamma_m = gamma_km / 1000.0  # 1/(W·m)

    alpha_db_per_km = 0.2  # typical
    alpha_m = (np.log(10.0) / 10.0) * alpha_db_per_km / 1000.0  # 1/m

    # ----------------------------
    # 5) Input powers (W) and phases (rad)
    # ----------------------------
    # Pumps: 0.5 W each; seeded signal: 1 mW; idler: 0 W
    p_in = np.array([0.5, 0.5, 1e-3, 0.0], dtype=float)
    phase_in = np.zeros(4, dtype=float)

    # ----------------------------
    # 6) Run scan and plot
    # ----------------------------
    plot_max_signal_gain_vs_lambda_signal(
        cfg=cfg,
        lambda_p1_m=lambda_p1,
        lambda_p2_m=lambda_p2,
        lambda_signal_m=lambda_signal,
        gamma=gamma_m,
        alpha=alpha_m,
        p_in=p_in,
        phase_in=phase_in,
        dispersion=disp,
        phase_matching_cfg=pm_cfg,
        length_unit="m",
        return_wavelength_unit="nm",
        save_path=None,
        show=True,
        gain_unit="db"
    )


if __name__ == "__main__":
    main_gain_spectrum()
