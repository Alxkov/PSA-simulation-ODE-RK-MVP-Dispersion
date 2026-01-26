from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import constants
from math import pi

from config import custom_simulation_config
from solver_fwm_sbs import SolverSettings
from simulation_sbs import run_single_simulation_sbs
from plotting_sbs import (
    plot_sbs_dashboard,
    plot_pump_depletion_and_stokes_growth,
    plot_forward_backward_powers_db,
    plot_acoustic_magnitude,
)
#
# if __name__ == '__main__':
#     print('Executing main')
#     # z, A = custom_seeded_signal()
#     # plot_powers(z, A)
#     # plot_signal_and_idler(z, A)
#     # plot_signal_and_idler_separate(z, A, title=" ")
#     # io_fwm.save_summary_csv("./summary.csv", z, A, overwrite=True)
#     scan_mismtach.scan_mismatch_seeded_signal(gain_mode="max")
#     print('Executed successfully')

def _omega_from_lambda_m(lam_m: float) -> float:
    # angular frequency ω = 2π c / λ
    return 2.0 * pi * constants.c / float(lam_m)

def main() -> None:
    # -----------------------------
    # Numerical configuration
    # -----------------------------
    cfg = custom_simulation_config(
        z_max=2.0e-1,        # km
        dz=1e-4,          # km  (0.1 m)
        save_every=20,
        check_nan=True,
        verbose=True,
    )

    solver_settings = SolverSettings(
        max_iter=60,
        tol_rel=1e-8,
        relax=0.5,
        init_B="zeros",
    )

    # -----------------------------
    # Optical frequencies
    # -----------------------------
    # Two pumps + seeded signal; idler computed by energy conservation:
    #   ω4 = ω1 + ω2 - ω3
    lam_p1 = 1540e-9
    lam_p2 = 1560e-9
    lam_s  = 1548e-9

    w1 = _omega_from_lambda_m(lam_p1)
    w2 = _omega_from_lambda_m(lam_p2)
    w3 = _omega_from_lambda_m(lam_s)
    w4 = w1 + w2 - w3

    omega = np.array([w1, w2, w3, w4], dtype=float)

    lam_i = 2.0 * pi * constants.c / w4
    print(f"[main] wavelengths (nm): pump1={lam_p1*1e9:.2f}, pump2={lam_p2*1e9:.2f}, "
          f"signal={lam_s*1e9:.2f}, idler≈{lam_i*1e9:.2f}")

    # -----------------------------
    # Fiber / Kerr parameters
    # -----------------------------
    gamma = 10.0   # 1/(W·km) (typical HNLF scale)
    alpha = 0.0    # 1/km (lossless for clarity)

    # Phase mismatch in your simplified model enters only via:
    #   Δβ = β3 + β4 - β1 - β2
    # Here we set Δβ ≈ 0 (near phase matching) by using equal betas.
    beta0 = 5.8e9
    beta = beta0 * np.ones(4, dtype=float)

    # -----------------------------
    # Forward boundary at z=0 (seed signal & idler)
    # -----------------------------
    # "Moderately strong pumps" + small seeded signal and idler:
    P_p1 = 0.8      # W
    P_p2 = 0.8      # W
    P_s  = 1e-3     # W
    P_i  = 1e-4     # W

    P_A0 = np.array([P_p1, P_p2, P_s, P_i], dtype=float)

    # Phases initial conditions
    phase_A0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # -----------------------------
    # Backward boundary at z=L (tiny seeds to "turn on" SBS)
    # -----------------------------
    # Without a spontaneous-noise model, SBS needs a seed. Use ultra-small powers.

    P_B_L = np.array([1e-12, 1e-12, 1e-12, 1e-12], dtype=float)
    phase_B_L = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # -----------------------------
    # SBS parameters (typical silica-ish magnitudes)
    # -----------------------------
    OmegaB = 2.0 * pi * 10.8e9   # rad/s  (Brillouin shift ~10.8 GHz)
    GammaB = 5e4                 # 1/s
    vA_km_s = 5.96e3 * 1e-3      # 5960 m/s -> 5.96 km/s

    # Coupling coefficients in your scalar model (project-specific scaling):
    # Choose small nonzero values so SBS is present but not overwhelmingly strong.
    kappa1 = 0.1
    kappa2 = 0.1

    # If None, simulation_sbs will use omega_B = omega - OmegaB
    # which makes dOmega = 0 by construction.
    omega_B = None

    # -----------------------------
    # Run simulation
    # -----------------------------
    z, A, B, Q, info = run_single_simulation_sbs(
        cfg,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        omega=omega,
        P_A0=P_A0,
        phase_A0=phase_A0,
        P_B_L=P_B_L,
        phase_B_L=phase_B_L,
        Q0=None,
        kappa1=kappa1,
        kappa2=kappa2,
        OmegaB=OmegaB,
        GammaB=GammaB,
        vA_km_s=vA_km_s,
        omega_B=omega_B,
        solver_settings=solver_settings,
    )

    print(f"[main] solver converged={info.get('converged')}  "
          f"iters={info.get('n_iter')}  rel_err_B={info.get('rel_err_B'):.3e}")

    # Quick end-of-fiber power summary
    P_out = np.abs(A[-1, :]) ** 2
    print("[main] Forward output powers |A|^2 at z=L (W):")
    print(f"        A1={P_out[0]:.6g}, A2={P_out[1]:.6g}, A3={P_out[2]:.6g}, A4={P_out[3]:.6g}")

    # -----------------------------
    # Plotting
    # -----------------------------
    fig1 = plot_sbs_dashboard(z, A, B, Q, title="FWM+SBS: moderately strong pumps + seeded signal & idler")
    fig2 = plot_forward_backward_powers_db(z, A, B, title="Forward/Backward powers (dB)")
    fig3 = plot_pump_depletion_and_stokes_growth(z, A, B, pump_index=0, title="Channel 1: forward vs backward (dB)")
    fig4 = plot_acoustic_magnitude(z, Q, title="Acoustic envelopes |Q|", squared=False)

    plt.show()

    # -----------------------------
    # Save results
    # -----------------------------
    np.savez(
        "result_fwm_sbs_seeded.npz",
        z=z,
        A=A,
        B=B,
        Q=Q,
        info=np.array([str(info)], dtype=object),
        omega=omega,
        beta=beta,
        P_A0=P_A0,
        P_B_L=P_B_L,
        gamma=np.array([gamma]),
        alpha=np.array([alpha]),
        OmegaB=np.array([OmegaB]),
        GammaB=np.array([GammaB]),
        vA_km_s=np.array([vA_km_s]),
        kappa1=np.array([kappa1]),
        kappa2=np.array([kappa2]),
    )
    print("[main] Saved: result_fwm_sbs_seeded.npz")


if __name__ == "__main__":
    main()

