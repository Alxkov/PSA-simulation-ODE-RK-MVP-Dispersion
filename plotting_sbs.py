"""
plotting_sbs.py

Matplotlib plotting utilities for coupled Scalar FWM + SBS simulations.

Expected arrays:
- z : shape (N,), increasing (km)
- A : shape (N,4), complex forward fields
- B : shape (N,4), complex backward fields (already aligned to increasing z)
- Q : shape (N,4), complex acoustic envelopes

Normalization assumed:
- Optical powers: |A_j|^2 and |B_j|^2 are in watts (consistent with your FWM project).
- Acoustic |Q| is dimensionless unless you define otherwise; we plot magnitude and magnitude^2.

This module is intentionally separate from plotting.py so that the pure-FWM plotting
remains unchanged.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_WAVE_LABELS = ("1", "2", "3", "4")


# ---------------------------------------------------------------------
# Validation and helpers
# ---------------------------------------------------------------------

def _ensure_1d(z: np.ndarray, name: str) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {z.shape}")
    if z.size < 2:
        raise ValueError(f"{name} must have at least 2 points, got {z.size}")
    return z


def _ensure_2d_complex(X: np.ndarray, name: str, nrows: int) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    if X.shape[0] != nrows:
        raise ValueError(f"{name} first dim must match z length {nrows}, got {X.shape[0]}")
    if X.shape[1] != 4:
        raise ValueError(f"{name} must have 4 columns (waves 1..4), got {X.shape[1]}")
    if not np.iscomplexobj(X):
        X = X.astype(np.complex128, copy=False)
    return X


def validate_sbs_arrays(
    z: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Q: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    z = _ensure_1d(z, "z")
    A = _ensure_2d_complex(A, "A", z.size)
    B = _ensure_2d_complex(B, "B", z.size)
    if Q is None:
        return z, A, B, None
    Q = _ensure_2d_complex(Q, "Q", z.size)
    return z, A, B, Q


def powers_from_fields(A: np.ndarray, B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return P=|A|^2 and S=|B|^2, both shape (N,4).
    """
    P = np.abs(A) ** 2
    S = np.abs(B) ** 2
    return P, S


def to_db(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    """
    Convert power-like quantity to dB with floor protection.
    """
    x = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(x, float(floor)))


def _labels(labels: Optional[Sequence[str]]) -> Tuple[str, str, str, str]:
    if labels is None:
        return DEFAULT_WAVE_LABELS
    if len(labels) != 4:
        raise ValueError("labels must have length 4")
    return tuple(str(s) for s in labels)  # type: ignore[return-value]


# ---------------------------------------------------------------------
# Main plots: forward/backward optical power
# ---------------------------------------------------------------------

def plot_forward_backward_powers(
    z: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    title: str = "Optical power evolution (linear scale)",
    show_total: bool = True,
) -> plt.Figure:
    """
    Plot |A_j|^2 and |B_j|^2 vs z (linear scale), with two subplots.

    Returns a Matplotlib Figure.
    """
    z, A, B, _ = validate_sbs_arrays(z, A, B, None)
    lab = _labels(labels)
    P, S = powers_from_fields(A, B)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1, ax2 = axes

    for j in range(4):
        ax1.plot(z, P[:, j], label=f"A{lab[j]}")
    if show_total:
        ax1.plot(z, np.sum(P, axis=1), linestyle="--", label="Σ|A|²")
    ax1.set_ylabel("Forward power |A|² [W]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    for j in range(4):
        ax2.plot(z, S[:, j], label=f"B{lab[j]}")
    if show_total:
        ax2.plot(z, np.sum(S, axis=1), linestyle="--", label="Σ|B|²")
    ax2.set_xlabel("z [km]")
    ax2.set_ylabel("Backward power |B|² [W]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_forward_backward_powers_db(
    z: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    title: str = "Optical power evolution (dB)",
    floor: float = 1e-30,
    show_total: bool = True,
) -> plt.Figure:
    """
    Plot 10log10(|A_j|^2) and 10log10(|B_j|^2) vs z in dB, with two subplots.
    """
    z, A, B, _ = validate_sbs_arrays(z, A, B, None)
    lab = _labels(labels)
    P, S = powers_from_fields(A, B)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1, ax2 = axes

    for j in range(4):
        ax1.plot(z, to_db(P[:, j], floor=floor), label=f"A{lab[j]}")
    if show_total:
        ax1.plot(z, to_db(np.sum(P, axis=1), floor=floor), linestyle="--", label="Σ|A|²")
    ax1.set_ylabel("Forward power [dBW]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    for j in range(4):
        ax2.plot(z, to_db(S[:, j], floor=floor), label=f"B{lab[j]}")
    if show_total:
        ax2.plot(z, to_db(np.sum(S, axis=1), floor=floor), linestyle="--", label="Σ|B|²")
    ax2.set_xlabel("z [km]")
    ax2.set_ylabel("Backward power [dBW]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------
# Acoustic plots
# ---------------------------------------------------------------------

def plot_acoustic_magnitude(
    z: np.ndarray,
    Q: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    title: str = "Acoustic envelope magnitude |Q|",
    squared: bool = False,
) -> plt.Figure:
    """
    Plot |Q_j| (or |Q_j|^2 if squared=True) vs z.
    """
    z = _ensure_1d(z, "z")
    Q = _ensure_2d_complex(Q, "Q", z.size)
    lab = _labels(labels)

    mag = np.abs(Q)
    y = mag ** 2 if squared else mag

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    for j in range(4):
        ax.plot(z, y[:, j], label=f"Q{lab[j]}")
    ax.set_xlabel("z [km]")
    ax.set_ylabel("|Q|² [arb.]" if squared else "|Q| [arb.]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_acoustic_phase(
    z: np.ndarray,
    Q: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    title: str = "Acoustic phase arg(Q)",
    unwrap: bool = True,
) -> plt.Figure:
    """
    Plot arg(Q_j) vs z. Optionally unwrap phase.
    """
    z = _ensure_1d(z, "z")
    Q = _ensure_2d_complex(Q, "Q", z.size)
    lab = _labels(labels)

    phase = np.angle(Q)
    if unwrap:
        phase = np.unwrap(phase, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    for j in range(4):
        ax.plot(z, phase[:, j], label=f"Q{lab[j]}")
    ax.set_xlabel("z [km]")
    ax.set_ylabel("Phase [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Derived diagnostics: gain, depletion, power exchange
# ---------------------------------------------------------------------

def plot_channel_gains_db(
    z: np.ndarray,
    A: np.ndarray,
    B: Optional[np.ndarray] = None,
    *,
    labels: Optional[Sequence[str]] = None,
    floor: float = 1e-30,
    title: str = "Per-channel gain relative to input",
    use_forward: bool = True,
    use_backward: bool = False,
) -> plt.Figure:
    """
    Plot per-channel gain in dB relative to z=0 for forward (A) and/or backward (B).

    Gain_j(z) = 10log10(P_j(z)/P_j(0))  (forward)
    Gain_j(z) = 10log10(S_j(z)/S_j(0))  (backward)

    Notes:
    - For backward waves, S_j(0) might be ~0, so gains can blow up; floor protects that.
    """
    z, A, B2, _ = validate_sbs_arrays(z, A, (B if B is not None else np.zeros_like(A)), None)
    lab = _labels(labels)

    P = np.abs(A) ** 2
    S = np.abs(B2) ** 2

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))

    if use_forward:
        P0 = np.maximum(P[0, :], floor)
        Gf = 10.0 * np.log10(np.maximum(P, floor) / P0[None, :])
        for j in range(4):
            ax.plot(z, Gf[:, j], label=f"A{lab[j]} gain")

    if use_backward:
        S0 = np.maximum(S[0, :], floor)
        Gb = 10.0 * np.log10(np.maximum(S, floor) / S0[None, :])
        for j in range(4):
            ax.plot(z, Gb[:, j], linestyle="--", label=f"B{lab[j]} gain")

    ax.set_xlabel("z [km]")
    ax.set_ylabel("Gain [dB]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_total_power_exchange(
    z: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    *,
    title: str = "Total forward/backward power vs z",
) -> plt.Figure:
    """
    Plot Σ|A|^2 and Σ|B|^2 vs z on the same axis.
    Useful to visualize SBS depletion/transfer trends.
    """
    z, A, B, _ = validate_sbs_arrays(z, A, B, None)
    P, S = powers_from_fields(A, B)

    Pf = np.sum(P, axis=1)
    Pb = np.sum(S, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(z, Pf, label="Σ|A|² (forward)")
    ax.plot(z, Pb, label="Σ|B|² (backward)")
    ax.set_xlabel("z [km]")
    ax.set_ylabel("Total power [W]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pump_depletion_and_stokes_growth(
    z: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    *,
    pump_index: int = 0,
    title: str = "Pump depletion and Stokes growth",
    floor: float = 1e-30,
) -> plt.Figure:
    """
    A convenience plot focused on one channel j (default j=1 -> index 0):
    - forward power |A_j|^2
    - backward power |B_j|^2
    both plotted in dB (dBW) to show many-decade changes.

    pump_index must be 0..3.
    """
    if pump_index not in (0, 1, 2, 3):
        raise ValueError("pump_index must be 0..3")

    z, A, B, _ = validate_sbs_arrays(z, A, B, None)
    P, S = powers_from_fields(A, B)

    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    ax.plot(z, to_db(P[:, pump_index], floor=floor), label=f"A{pump_index+1} [dBW]")
    ax.plot(z, to_db(S[:, pump_index], floor=floor), label=f"B{pump_index+1} [dBW]")
    ax.set_xlabel("z [km]")
    ax.set_ylabel("Power [dBW]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Quick "one-call" dashboard
# ---------------------------------------------------------------------

def plot_sbs_dashboard(
    z: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    *,
    labels: Optional[Sequence[str]] = None,
    floor: float = 1e-30,
    title: str = "FWM+SBS dashboard",
) -> plt.Figure:
    """
    2x2 dashboard:
    (1) forward powers (dB)
    (2) backward powers (dB)
    (3) |Q| (linear)
    (4) total forward/backward power (linear)

    Returns a single Figure.
    """
    z, A, B, Q = validate_sbs_arrays(z, A, B, Q)
    assert Q is not None
    lab = _labels(labels)

    P, S = powers_from_fields(A, B)
    Pf = np.sum(P, axis=1)
    Pb = np.sum(S, axis=1)
    Qm = np.abs(Q)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax11, ax12 = axes[0]
    ax21, ax22 = axes[1]

    # Forward dB
    for j in range(4):
        ax11.plot(z, to_db(P[:, j], floor=floor), label=f"A{lab[j]}")
    ax11.set_ylabel("Forward [dBW]")
    ax11.grid(True, alpha=0.3)
    ax11.legend(loc="best")

    # Backward dB
    for j in range(4):
        ax12.plot(z, to_db(S[:, j], floor=floor), label=f"B{lab[j]}")
    ax12.set_ylabel("Backward [dBW]")
    ax12.grid(True, alpha=0.3)
    ax12.legend(loc="best")

    # |Q|
    for j in range(4):
        ax21.plot(z, Qm[:, j], label=f"Q{lab[j]}")
    ax21.set_xlabel("z [km]")
    ax21.set_ylabel("|Q| [arb.]")
    ax21.grid(True, alpha=0.3)
    ax21.legend(loc="best")

    # Totals
    ax22.plot(z, Pf, label="Σ|A|²")
    ax22.plot(z, Pb, label="Σ|B|²")
    ax22.set_xlabel("z [km]")
    ax22.set_ylabel("Total power [W]")
    ax22.grid(True, alpha=0.3)
    ax22.legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
