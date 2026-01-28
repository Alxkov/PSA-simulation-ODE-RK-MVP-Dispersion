# main.py


from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import constants
from math import pi

from config import custom_simulation_config
from solver_fwm_sbs import SolverSettings


def _omega_from_lambda_m(lam_m: float) -> float:
    # ω = 2π c / λ
    return 2.0 * pi * constants.c / float(lam_m)


def _try_import_new_runner():
    """
    Prefer the new simulation runner API if it exists:
      run_single_simulation_fwm_sbs(cfg, ..., solver_settings=..., forward_method=...)

    Otherwise fall back to the earlier API:
      run_single_simulation_sbs(cfg, ..., solver_settings=...)
    """
    try:
        from simulation_sbs import run_single_simulation_fwm_sbs as run_new
        return "new", run_new
    except Exception:
        from simulation_sbs import run_single_simulation_sbs as run_old
        return "old", run_old


def main() -> None:
    # -----------------------------
    # Numerical config
    # -----------------------------
    cfg = custom_simulation_config(
        z_max=1.0,        # km
        dz=8e-4,          # km (1 m) -- expQ should tolerate this much better than full RK4 on Q
        save_every=10,
        check_nan=True,
        verbose=True,
    )

    # Use the new stiff-friendly method by default
    solver_settings = SolverSettings(
        max_iter=60,
        tol_rel=1e-8,
        relax=0.5,
        init_B="zeros",
        forward_method="expQ",   # <-- key
        q_forcing="avg",
        lambda_eps=1e-14,
    )

    # -----------------------------
    # Define wavelengths and frequencies
    # -----------------------------
    lam_p1 = 1550e-9
    lam_p2 = 1560e-9
    lam_s  = 1540e-9

    w1 = _omega_from_lambda_m(lam_p1)
    w2 = _omega_from_lambda_m(lam_p2)
    w3 = _omega_from_lambda_m(lam_s)
    w4 = w1 + w2 - w3  # idler by energy conservation

    omega = np.array([w1, w2, w3, w4], dtype=float)
    lam_i = 2.0 * pi * constants.c / w4

    print(
        "[main] wavelengths (nm): "
        f"pump1={lam_p1*1e9:.2f}, pump2={lam_p2*1e9:.2f}, "
        f"signal={lam_s*1e9:.2f}, idler≈{lam_i*1e9:.2f}"
    )

    # -----------------------------
    # Fiber parameters (demo values)
    # -----------------------------
    gamma = 10.0   # 1/(W·km)
    alpha = 0.0    # 1/km

    # Keep Δβ ≈ 0 in this simplified model by using equal betas
    beta0 = 5.8e9
    beta = beta0 * np.ones(4, dtype=float)

    # -----------------------------
    # Inputs: moderately strong pumps + seeded signal/idler
    # -----------------------------
    P_p1 = 1      # W
    P_p2 = 1      # W
    P_s  = 1e-3     # W
    P_i  = 1e-4     # W
    P_A0 = np.array([P_p1, P_p2, P_s, P_i], dtype=float)

    phase_A0 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # Backward seeds at z=L (tiny; needed without spontaneous noise model)
    P_B_L = np.array([1e-3, 1e-3, 1e-3, 1e-3], dtype=float)
    phase_B_L = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # -----------------------------
    # SBS parameters (typical silica-ish scales)
    # -----------------------------
    OmegaB = 2.0 * pi * 10.8e9   # rad/s
    GammaB = 1e8     # 1/s
    vA_km_s = 5.96e3 * 1e-3      # 5960 m/s -> 5.96 km/s

    # scalar coupling parameters (model scaling dependent)
    kappa1 = 5e5
    kappa2 = 5e6

    # if omega_B is None, the parameter factory uses omega_B = omega - OmegaB (=> dOmega=0)
    omega_B = None

    # -----------------------------
    # Run (new API preferred; old API supported)
    # -----------------------------
    api_kind, run_sim = _try_import_new_runner()
    print(f"[main] simulation_sbs API detected: {api_kind}")

    if api_kind == "new":
        # New runner: run_single_simulation_fwm_sbs
        z, A, B, Q, info = run_sim(
            cfg,
            gamma=gamma,
            alpha=alpha,
            beta=beta,
            omega=omega,
            p_in_forward=P_A0,
            p_in_backward=P_B_L,
            phase_in_forward=phase_A0,
            phase_in_backward=phase_B_L,
            kappa1=kappa1,
            kappa2=kappa2,
            v_a=vA_km_s,
            Gamma_B=GammaB,
            Omega_B=OmegaB,
            delta_Omega=np.zeros(4, dtype=float),
            q0=None,
            solver_settings=solver_settings,
            forward_method="expQ",
        )
    else:
        # Old runner: run_single_simulation_sbs (power/phase interface)
        z, A, B, Q, info = run_sim(
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

    print(
        f"[main] solver converged={info.get('converged')}  "
        f"iters={info.get('n_iter')}  rel_err_B={info.get('rel_err_B'):.3e}  "
        f"forward_method={info.get('forward_method')}"
    )

    P_out = np.abs(A[-1, :]) ** 2
    print("[main] Forward output powers |A|^2 at z=L (W):")
    print(f"        A1={P_out[0]:.6g}, A2={P_out[1]:.6g}, A3={P_out[2]:.6g}, A4={P_out[3]:.6g}")

    # -----------------------------
    # Plotting (optional but useful)
    # -----------------------------
    try:
        from plotting_sbs import plot_sbs_dashboard
        fig = plot_sbs_dashboard(z, A, B, Q, title="FWM+SBS (expQ forward method): seeded signal & idler")
        plt.show()
    except Exception as e:
        print(f"[main] plotting_sbs not available or failed: {e}")

    # -----------------------------
    # Save results
    # -----------------------------
    np.savez(
        "result_fwm_sbs_expQ_seeded.npz",
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
    print("[main] Saved: result_fwm_sbs_expQ_seeded.npz")


if __name__ == "__main__":
    main()
