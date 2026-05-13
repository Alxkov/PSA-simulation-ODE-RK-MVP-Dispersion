# main.py
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

import plotting
from config import custom_simulation_config
from dispersion import dispersion_params_from_D_S, delta_beta_from_omegas, delta_beta_symmetric
from frequency_plan import (
    plan_from_wavelengths,
    infer_symmetry_from_omegas,
    lambda_from_omega,
    describe_plan,
)
from phase_matching import PhaseMatchingConfig, PhaseMatchingMethod, compute_phase_mismatch
from simulation import run_single_simulation
from scan_mismtach import (plot_max_signal_gain_vs_lambda_signal,
                           plot_dbeta_vs_lambda_signal,
                           plot_max_gain_and_dbeta_vs_lambda_signal)


def main_single_simulation() -> None:
    # ----------------------------
    # 1) Numerical grid (meters)
    # ----------------------------
    # Fiber length 500 m, step 0.1 m

    cfg = custom_simulation_config(z_max=400.0, dz=0.2)

    # ----------------------------
    # 2) Frequency plan (dual-pump)
    #    Order: [pump1, pump2, signal, idler]
    # ----------------------------
    lambda1 = 1545e-9  # pump1
    lambda2 = 1555e-9  # pump2
    lambda3 = 1530e-9  # signal (m)
    omega = plan_from_wavelengths(lambda1, lambda2, lambda3, lambda4_m=None)

    # (Optional) print the plan for sanity
    print(describe_plan(omega))

    # Infer symmetric variables to define ωc (useful as dispersion expansion point)
    sp = infer_symmetry_from_omegas(omega1=omega[0], omega2=omega[1], omega3=omega[2], omega4=omega[3])
    lambda_c = lambda_from_omega(sp.omega_c)

    disp = dispersion_params_from_D_S(
        lambda_ref_m=lambda_c,
        D=-0.1,
        S=0.02,
        dSdlmbd=1e-8,
        D_units="ps/nm/km",
        S_units="ps/nm^2/km",
        dSdlmbd_units="ps/nm^3/km",
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
    gamma_km = 11.5  # 1/(W·km)
    gamma_m = gamma_km / 1000.0  # 1/(W·m)

    alpha_db_per_km = 0.01  # dB/km (power loss)
    alpha_m = (np.log(10.0) / 10.0) * alpha_db_per_km / 1000.0  # 1/m

    # ----------------------------
    # 5) Inputs
    # ----------------------------
    p_in = np.array([1.0, 1.0, 1e-3, 1e-10], dtype=float)  # W
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

    plotting.plot_fwm_sbs_powers_forward(z, A, scale="dbW", colors=("red", "orange", "blue", "lightseagreen"))

def main_gain_spectrum():
    lambda_p1 = 1540e-9  # pump1
    lambda_p2 = 1560e-9  # pump2

    lambda_pc = (lambda_p1 + lambda_p2) / 2  # auxiliary for sweeping convenience

    # Signal scan: 1520..1580 nm
    lambda_signal = np.linspace(lambda_p1 - (lambda_pc - lambda_p1) * 0.8,
                                lambda_p2 + (lambda_p2 - lambda_pc) * 0.8, 50)  # 1 nm step
    lambda_signal = np.linspace(1500e-9,
                                1600e-9, 50)  # 1 nm step

    # ----------------------------
    # 2) Simulation grid (meters)
    # ----------------------------
    # 200 m fiber with dz=0.2 m -> 1000 steps per run.
    cfg = custom_simulation_config(z_max=500.0, dz=0.1)
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
        D=-0.05,
        S=0.02,
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
    gamma_km = 11.5  # 1/(W·km)
    gamma_m = gamma_km / 1000.0  # 1/(W·m)

    alpha_db_per_km = 0.5  # typical
    alpha_m = (np.log(10.0) / 10.0) * alpha_db_per_km / 1000.0  # 1/m

    # ----------------------------
    # 5) Input powers (W) and phases (rad)
    # ----------------------------
    p_in = np.array([1.0, 1.0, 1e-6, 1e-10], dtype=float)
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


def main_gain_spectrum_dbeta():
    lambda_p1 = 1545e-9  # pump1
    lambda_p2 = 1555e-9  # pump2


    lambda_pc = (lambda_p1 + lambda_p2)/2  # auxiliary for sweeping convenience

    # Signal scan: 1520..1580 nm
    lambda_signal = np.linspace(lambda_p1 - (lambda_pc - lambda_p1) * 0.8,
                                lambda_p2 + (lambda_p2 - lambda_pc) * 0.8, 50)  # 1 nm step
    lambda_signal = np.linspace(1500e-9,
                                1600e-9, 50)  # 1 nm step

    # ----------------------------
    # 2) Simulation grid (meters)
    # ----------------------------
    # 200 m fiber with dz=0.2 m -> 1000 steps per run.
    cfg = custom_simulation_config(z_max=500.0, dz=0.1)

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
        D=1.0,
        S=0.02,
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
    gamma_km = 20  # 1/(W·km)
    gamma_m = gamma_km / 1000.0  # 1/(W·m)

    alpha_db_per_km = 0.8  # typical
    alpha_m = (np.log(10.0) / 10.0) * alpha_db_per_km / 1000.0  # 1/m

    # ----------------------------
    # 5) Input powers (W) and phases (rad)

    p_in = np.array([1.0, 1.0, 1e-10, 1e-12], dtype=float)
    phase_in = np.zeros(4, dtype=float)
    x, gmax, db = plot_max_gain_and_dbeta_vs_lambda_signal(
        cfg=cfg,
        lambda_p1_m=lambda_p1,
        lambda_p2_m=lambda_p2,
        lambda_signal_m=lambda_signal,
        gamma=gamma_m,
        alpha=alpha_m,
        p_in=p_in,
        dispersion=disp,
        length_unit="m",
        gain_unit="dB",
        phase_in=phase_in
    )


def main_gain_spectrum_dbeta_twin_axis() -> None:
    """
    Plot a dual-pump FWM signal-wavelength gain spectrum with phase mismatch on a twin y-axis.

    This is the same style as the GNLSE_SSFM wavelength-sweep plot:
      - left y-axis: signal gain and idler conversion in dB;
      - right y-axis: phase mismatch dBeta(lambda_s);
      - title: pump wavelengths, fiber length, dz, gamma, alpha, and dispersion parameters;
      - no highlighted maximum-gain point.

    Wave order throughout the project is:
        [pump1, pump2, signal, idler] = [1, 2, 3, 4].
    """
    # ----------------------------
    # 1) Pump wavelengths and signal scan
    # ----------------------------
    lambda_p1 = 1548e-9  # pump1 [m]
    lambda_p2 = 1552e-9  # pump2 [m]
    lambda_signal = np.linspace(1500.0e-9, 1600e-9, 50)  # signal wavelength sweep [m]

    # ----------------------------
    # 2) Simulation grid (meters)
    # ----------------------------
    cfg = custom_simulation_config(z_max=200.0, dz=0.2)

    # ----------------------------
    # 3) Dispersion reference at omega_c
    # ----------------------------
    omega_ref = plan_from_wavelengths(lambda_p1, lambda_p2, float(lambda_signal[0]), lambda4_m=None)
    sp = infer_symmetry_from_omegas(
        omega1=omega_ref[0],
        omega2=omega_ref[1],
        omega3=omega_ref[2],
        omega4=omega_ref[3],
    )
    lambda_c = lambda_from_omega(sp.omega_c)

    D_ps_nm_km = 0.1
    S_ps_nm2_km = 0.02
    dSdlmbd_ps_nm3_km = 1e-5

    disp = dispersion_params_from_D_S(
        lambda_ref_m=lambda_c,
        D=D_ps_nm_km,
        S=S_ps_nm2_km,
        dSdlmbd=dSdlmbd_ps_nm3_km,
        D_units="ps/nm/km",
        S_units="ps/nm^2/km",
        dSdlmbd_units="ps/nm^3/km",
        omega_ref=sp.omega_c,
    )

    pm_cfg = PhaseMatchingConfig(
        method=PhaseMatchingMethod.SYMMETRIC_EVEN,
        even_orders=(2, 4),
        max_order=4,
        atol=0.0,
        rtol=1e-12,
        provided_delta_beta=None,
    )

    # ----------------------------
    # 4) Fiber nonlinearity and loss
    # ----------------------------
    gamma_km = 11.5  # [1/(W km)]
    gamma_m = gamma_km / 1000.0  # [1/(W m)]

    alpha_db_per_km = 0.8  # [dB/km]
    alpha_m = (np.log(10.0) / 10.0) * alpha_db_per_km / 1000.0  # [1/m]

    # ----------------------------
    # 5) Input powers and phases
    # ----------------------------
    p_in = np.array([1.0, 1.0, 1e-12, 1e-15], dtype=float)  # [W]
    phase_in = np.zeros(4, dtype=float)  # [rad]
    nonlinear_phase_shift_ref = gamma_m * float(p_in[0] + p_in[1])  # [1/m]
    nonlinear_mismatch_ref = -nonlinear_phase_shift_ref  # [1/m]

    signal_gain_db = np.full(lambda_signal.shape, np.nan, dtype=float)
    idler_conversion_db = np.full(lambda_signal.shape, np.nan, dtype=float)
    dbeta = np.full(lambda_signal.shape, np.nan, dtype=float)

    print("=== Starting FWM_CW_Dispersion gain-spectrum sweep ===")
    print(f"Signal wavelength range: {lambda_signal[0] * 1e9:.3f}..{lambda_signal[-1] * 1e9:.3f} nm")
    print(f"Number of points: {lambda_signal.size}")
    print(f"gamma(P1 + P2): {nonlinear_phase_shift_ref:.6e} 1/m")
    print(f"-gamma(P1 + P2): {nonlinear_mismatch_ref:.6e} 1/m")

    iterator = range(lambda_signal.size)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Gain spectrum", total=lambda_signal.size)

    for i in iterator:
        lam_s = float(lambda_signal[i])
        omega = plan_from_wavelengths(lambda_p1, lambda_p2, lam_s, lambda4_m=None)

        pm_res = compute_phase_mismatch(
            omegas=omega,
            disp=disp,
            cfg=pm_cfg,
            symmetric_hint=None,
        )
        dbeta[i] = float(pm_res.delta_beta)

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

        powers = np.abs(A) ** 2
        # signal_metric_power = float(np.max(powers[:, 2]))
        signal_metric_power = float(powers[-1,2])
        # idler_metric_power = float(np.max(powers[:, 3]))
        idler_metric_power = float(powers[-1,3])

        signal_gain_db[i] = 10.0 * np.log10(signal_metric_power / p_in[2])
        idler_conversion_db[i] = 10.0 * np.log10(idler_metric_power / p_in[2])

        if tqdm is not None and hasattr(iterator, "set_postfix_str"):
            iterator.set_postfix_str(
                f"lambda_s={lam_s * 1e9:.3f} nm, "
                f"G_s={signal_gain_db[i]:+.2f} dB, "
                f"dBeta={dbeta[i]:+.3e} 1/m"
            )
        elif tqdm is None:
            print(
                f"[{i + 1:03d}/{lambda_signal.size:03d}] "
                f"lambda_s={lam_s * 1e9:10.6f} nm, "
                f"G_s={signal_gain_db[i]:+9.3f} dB, "
                f"CE_i={idler_conversion_db[i]:+9.3f} dB, "
                f"dBeta={dbeta[i]:+.6e} 1/m"
            )

    x_nm = lambda_signal * 1e9
    pump1_nm = lambda_p1 * 1e9
    pump2_nm = lambda_p2 * 1e9
    kappa = dbeta + nonlinear_phase_shift_ref  # kappa(lambda_s) = Delta beta(lambda_s) + gamma(P1 + P2) [1/m]

    # ------------------------------------------------------------------
    # Plot 1: spectrum + dBeta twin axis + pump wavelength markers
    # ------------------------------------------------------------------
    fig_dbeta, ax_gain = plt.subplots(figsize=(9.0, 5.0))
    ax_gain.plot(x_nm, idler_conversion_db, marker="s", linewidth=1.5, label="idler conversion", color="orange")
    ax_gain.plot(x_nm, signal_gain_db, marker="o", linewidth=1.5, label="signal gain", color="blue")
    ax_gain.axvline(pump1_nm, linestyle="--", linewidth=1.2, alpha=0.75, label=r"pump 1", color="red")
    ax_gain.axvline(pump2_nm, linestyle="--", linewidth=1.2, alpha=0.75, label=r"pump 2", color="violet")
    ax_gain.set_xlabel(r"signal wavelength $\lambda_s$ [nm]")
    ax_gain.set_ylabel("gain / conversion [dB]")
    ax_gain.grid(True, which="both", alpha=0.35)
    ax_gain.legend(loc="best")

    ax_dbeta = ax_gain.twinx()
    ax_dbeta.plot(x_nm, dbeta, linestyle="--", linewidth=1.4, label=r"$\Delta\beta$")
    ax_dbeta.axhline(
        nonlinear_mismatch_ref,
        linestyle=":",
        linewidth=1.6,
        label=r"-$\gamma(P_{p1}+P_{p2})$",
    )
    ax_dbeta.set_ylabel(r"$\Delta\beta = \beta_s + \beta_i - \beta_{p1} - \beta_{p2}$ [1/m]")
    ax_dbeta.legend(loc="best")

    fig_dbeta.suptitle(
        "Dual-pump CW FWM gain spectrum\n"
        f"pumps: {pump1_nm:.3f} nm, {pump2_nm:.3f} nm; "
        f"L={cfg.z_max:.3f} m, dz={cfg.dz:.3f} m; "
        f"gamma={gamma_km:.3f} 1/(W km) \n"
        f"D={D_ps_nm_km:.3f} ps/(nm km), S={S_ps_nm2_km:.3f} ps/(nm^2 km), P1={p_in[0]:.3f} W, P2={p_in[1]:.3f} W"
    )
    fig_dbeta.tight_layout(rect=(0.0, 0.0, 1.0, 1.01))

    # ------------------------------------------------------------------
    # Plot 2: clean spectrum only.
    # No dBeta curve, no pump-power line, and no pump wavelength markers.
    # ------------------------------------------------------------------
    fig_spectrum, ax_spectrum = plt.subplots(figsize=(9.0, 5.0))
    ax_spectrum.plot(x_nm, idler_conversion_db, marker="s", linewidth=1.5, label="idler conversion", color="orange")
    ax_spectrum.plot(x_nm, signal_gain_db, marker="o", linewidth=1.5, label="signal gain", color="blue")
    ax_spectrum.set_xlabel(r"signal wavelength $\lambda_s$ [nm]")
    ax_spectrum.set_ylabel("gain / conversion [dB]")
    ax_spectrum.grid(True, which="both", alpha=0.35)
    ax_spectrum.legend(loc="best")
    fig_spectrum.suptitle(
        "Dual-pump CW FWM gain spectrum\n"
        f"pumps: {pump1_nm:.3f} nm, {pump2_nm:.3f} nm; "
        f"L={cfg.z_max:.3f} m, dz={cfg.dz:.3f} m; "
        f"L={cfg.z_max:.3f} m, dz={cfg.dz:.3f} m; "
        f"gamma={gamma_km:.3f} 1/(W km) \n"
        f"D={D_ps_nm_km:.3f} ps/(nm km), S={S_ps_nm2_km:.3f} ps/(nm^2 km), P1={p_in[0]:.3f} W, P2={p_in[1]:.3f} W"
    )
    fig_spectrum.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    # ------------------------------------------------------------------
    # Plot 3: spectrum + kappa twin axis + pump wavelength markers.
    # Here kappa(lambda_s) = Delta beta(lambda_s) + gamma(P1 + P2).
    # The horizontal zero line marks the effective phase-matching condition
    # kappa = 0, i.e. Delta beta = -gamma(P1 + P2).
    # ------------------------------------------------------------------
    fig_kappa, ax_gain_kappa = plt.subplots(figsize=(9.0, 5.0))
    ax_gain_kappa.plot(x_nm, idler_conversion_db, marker="s", linewidth=1.5, label="idler conversion",
                       color='orange')
    ax_gain_kappa.plot(x_nm, signal_gain_db, marker="o", linewidth=1.5, label="signal gain", color='blue')
    ax_gain_kappa.axvline(pump1_nm, linestyle="--", linewidth=1.2, alpha=0.75, label=r"pump 1", color="red")
    ax_gain_kappa.axvline(pump2_nm, linestyle="--", linewidth=1.2, alpha=0.75, label=r"pump 2", color="violet")
    ax_gain_kappa.set_xlabel(r"signal wavelength $\lambda_s$ [nm]")
    ax_gain_kappa.set_ylabel("gain / conversion [dB]")
    ax_gain_kappa.grid(True, which="both", alpha=0.35)
    ax_gain_kappa.legend(loc="best")

    ax_kappa = ax_gain_kappa.twinx()
    ax_kappa.plot(
        x_nm,
        kappa,
        linestyle="--",
        linewidth=1.4,
        label=r"$\kappa = \Delta\beta + \gamma(P_{p1}+P_{p2})$"
    )
    ax_kappa.axhline(
        0.0,
        linestyle=":",
        linewidth=1.6,
        label=r"$\kappa = 0$"
    )
    ax_kappa.set_ylabel(r"$\kappa = \Delta\beta + \gamma(P_{p1}+P_{p2})$ [1/m]")
    ax_kappa.legend(loc="best")

    fig_kappa.suptitle(
        "Dual-pump CW FWM gain spectrum with effective mismatch\n"
        f"pumps: {pump1_nm:.3f} nm, {pump2_nm:.3f} nm; "
        f"L={cfg.z_max:.3f} m, dz={cfg.dz:.3f} m; "
        f"gamma={gamma_km:.3f} 1/(W km) \n"
        f"D={D_ps_nm_km:.3f} ps/(nm km), S={S_ps_nm2_km:.3f} ps/(nm^2 km), P1={p_in[0]:.3f} W, P2={p_in[1]:.3f} W"
    )
    fig_kappa.tight_layout(rect=(0.0, 0.0, 1.0, 1.01))

    plt.show()


if __name__ == "__main__":
    # main_single_simulation()
    main_gain_spectrum_dbeta_twin_axis()
