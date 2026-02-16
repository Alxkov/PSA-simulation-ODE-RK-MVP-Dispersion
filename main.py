# main.py
from __future__ import annotations

import numpy as np

import plotting
from config import custom_simulation_config
from dispersion import dispersion_params_from_D_S, delta_beta_from_omegas
from frequency_plan import (
    plan_from_wavelengths,
    infer_symmetry_from_omegas,
    lambda_from_omega,
    describe_plan,
)
from phase_matching import PhaseMatchingConfig, PhaseMatchingMethod
from simulation import run_single_simulation


def main() -> None:
    # ----------------------------
    # 1) Numerical grid (meters)
    # ----------------------------
    # Fiber length 500 m, step 0.1 m
    cfg = custom_simulation_config(z_max=500.0, dz=0.1)

    # ----------------------------
    # 2) Frequency plan (dual-pump)
    #    Order: [pump1, pump2, signal, idler]
    # ----------------------------
    lambda1 = 1545e-9  # pump1
    lambda2 = 1555e-9  # pump2
    lambda3 = 1590e-9  # signal
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
        omega_ref=sp.omega_c,
        beta4=0.0,
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

    alpha_db_per_km = 0.2  # dB/km (power loss)
    alpha_m = (np.log(10.0) / 10.0) * alpha_db_per_km / 1000.0  # 1/m

    # ----------------------------
    # 5) Inputs
    # ----------------------------
    p_in = np.array([0.5, 0.5, 1e-3, 0.0], dtype=float)  # W
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

    print("\n--- Results ---")
    print(f"z_end = {z[-1]:.3f} m")
    print(f"P_in  [W] = {p_in}")
    print(f"P_out [W] = {P_out}")
    print(f"Signal gain = {gain_signal_db:.3f} dB")
    print(f"db = {db:.3f}")
    print(f"gamma(P1 + P2) = {gamma_km * (p_in[0] + p_in[1]):.3f} km^-1")

    plotting.plot_fwm_sbs_powers_forward(z, A, scale="dbW")


if __name__ == "__main__":
    main()
