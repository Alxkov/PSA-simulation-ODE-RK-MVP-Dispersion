"""
plotting.py

Visualization utilities for simulation results.

This file originally assumed A has shape (N,4).
It is now generalized to support:
- the original 4-wave FWM outputs A(z) with shape (N,4)
- the extended FWM+SBS outputs A(z), B(z), Q(z) each with shape (N,4)

Backward compatibility:
- Existing functions (plot_powers, plot_signal_and_idler, etc.) keep their signatures
  (only adding optional parameters with defaults where needed).
"""

from __future__ import annotations

from typing import Sequence, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Internal validation helpers
# -----------------------------

def _validate_z(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim != 1:
        raise ValueError("z must be a 1D array")
    if z_arr.size < 2:
        raise ValueError("z must contain at least 2 points")
    return z_arr


def _validate_z_Y(z: np.ndarray, Y: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    z_arr = _validate_z(z)
    Y_arr = np.asarray(Y)

    if Y_arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array with shape (N, M)")

    if Y_arr.shape[0] != z_arr.shape[0]:
        raise ValueError(f"{name}.shape[0] must match z.shape[0]")

    if not np.iscomplexobj(Y_arr):
        Y_arr = Y_arr.astype(np.complex128, copy=False)
    else:
        Y_arr = Y_arr.astype(np.complex128, copy=False)

    return z_arr, Y_arr


def _validate_z_A(z: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_arr, A_arr = _validate_z_Y(z, A, name="A")
    if A_arr.shape[1] != 4:
        raise ValueError("A must have shape (N, 4)")
    return z_arr, A_arr


def _validate_labels(labels: Sequence[str], n: int, *, name: str = "labels") -> tuple[str, ...]:
    if len(labels) != n:
        raise ValueError(f"{name} must have length {n}")
    return tuple(labels)


def _apply_log_clip(y: np.ndarray, *, eps: float) -> np.ndarray:
    if eps <= 0.0:
        raise ValueError("eps must be > 0 for log clipping")
    return np.maximum(y, eps)


def _to_db10(y: np.ndarray, *, eps: float) -> np.ndarray:
    y_safe = _apply_log_clip(y, eps=eps)
    return 10.0 * np.log10(y_safe)


# -----------------------------
# One plotting "engine"
# -----------------------------

def _plot_series(
    z: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    *,
    title: Optional[str],
    xlabel: str,
    ylabel: str,
    yscale: str,          # "linear" | "log"
    log_base: float,      # 10 or np.e (only used if yscale=="log")
    show: bool,
    save_path: Optional[str],
    figsize: Tuple[float, float] = (8.0, 5.0),
) -> None:
    z_arr = _validate_z(z)
    y_arr = np.asarray(y, dtype=float)

    if y_arr.ndim != 2:
        raise ValueError("y must be a 2D array with shape (N, M)")
    if y_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("y.shape[0] must match z.shape[0]")
    if y_arr.shape[1] != len(labels):
        raise ValueError("labels length must match y.shape[1]")

    plt.figure(figsize=figsize)

    for j, lab in enumerate(labels):
        plt.plot(z_arr, y_arr[:, j], label=lab)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if yscale == "log":
        if log_base == 10:
            plt.yscale("log", base=10)
        elif log_base == np.e:
            plt.yscale("log", base=np.e)
        else:
            raise ValueError("log_base must be 10 or np.e when yscale='log'")
    elif yscale != "linear":
        raise ValueError("yscale must be 'linear' or 'log'")

    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()


# -----------------------------
# Generic public helpers
# -----------------------------

def plot_abs_matrix(
    z: np.ndarray,
    Y: np.ndarray,
    *,
    labels: Sequence[str],
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    z_unit: str = "m",
    ylabel: str = "|Y(z)|",
) -> None:
    z_arr, Y_arr = _validate_z_Y(z, Y, name="Y")
    labels_t = _validate_labels(labels, Y_arr.shape[1], name="labels")

    absY = np.abs(Y_arr)

    _plot_series(
        z_arr,
        absY,
        labels_t,
        title=title,
        xlabel=f"z [{z_unit}]",
        ylabel=ylabel,
        yscale="linear",
        log_base=10,
        show=show,
        save_path=save_path,
    )


def plot_power_matrix(
    z: np.ndarray,
    Y: np.ndarray,
    *,
    labels: Sequence[str],
    title: Optional[str] = None,
    scale: str = "linear",      # "linear" | "log" | "dbW"
    log_base: float = 10,
    eps: float = 1e-30,
    show: bool = True,
    save_path: Optional[str] = None,
    z_unit: str = "m",
    ylabel_linear: str = "P(z) [W]",
    ylabel_db: str = "P(z) [dBW]",
) -> None:
    """
    Plot power-like quantity |Y|^2.

    scale:
      - "linear": plot |Y|^2 linearly
      - "log":    plot |Y|^2 on logarithmic y-axis (base 10 or e)
      - "dbW":    plot 10*log10(|Y|^2 / 1W) (numerically: 10*log10(|Y|^2))
    """
    z_arr, Y_arr = _validate_z_Y(z, Y, name="Y")
    labels_t = _validate_labels(labels, Y_arr.shape[1], name="labels")

    P = np.abs(Y_arr) ** 2

    if scale == "linear":
        _plot_series(
            z_arr,
            P,
            labels_t,
            title=title,
            xlabel=f"z [{z_unit}]",
            ylabel=ylabel_linear,
            yscale="linear",
            log_base=log_base,
            show=show,
            save_path=save_path,
        )
        return

    if scale == "log":
        P_plot = _apply_log_clip(P, eps=eps)
        _plot_series(
            z_arr,
            P_plot,
            labels_t,
            title=title,
            xlabel=f"z [{z_unit}]",
            ylabel=ylabel_linear,
            yscale="log",
            log_base=log_base,
            show=show,
            save_path=save_path,
        )
        return

    if scale == "dbW":
        P_db = _to_db10(P, eps=eps)
        _plot_series(
            z_arr,
            P_db,
            labels_t,
            title=title,
            xlabel=f"z [{z_unit}]",
            ylabel=ylabel_db,
            yscale="linear",
            log_base=log_base,
            show=show,
            save_path=save_path,
        )
        return

    raise ValueError("scale must be one of: 'linear', 'log', 'dbW'")


def plot_total_powers_AB(
    z: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    *,
    title: str | None = "Total optical powers in A and B waves",
    scale: str = "linear",        # "linear" | "log" | "dbW"
    log_base: float = 10.0,
    eps: float = 1e-30,
    show: bool = True,
    save_path: str | None = None,
    z_unit: str = "m",
    figsize: tuple[float, float] = (8.0, 5.0),
) -> None:
    """
    Plot total powers in forward (A) and backward (B) waves versus z.

    Definitions:
        P_A_tot(z) = sum_j |A_j(z)|^2
        P_B_tot(z) = sum_j |B_j(z)|^2

    Parameters
    ----------
    z : (N,) array
        Propagation coordinate.
    A : (N,4) complex array
        Forward waves.
    B : (N,4) complex array
        Backward waves.
    title : optional str
        Figure title.
    scale : "linear" | "log" | "dbW"
        - "linear": plot powers in W on linear axis
        - "log": plot powers in W on log y-axis (uses clipping with eps)
        - "dbW": plot 10*log10(P[W]) on linear y-axis (uses clipping with eps)
    log_base : float
        Base for log axis if scale=="log" (10 or np.e recommended).
    eps : float
        Minimum power used for clipping in log/dbW operations.
    show : bool
        Whether to show the plot.
    save_path : optional str
        If given, save the figure to this path.
    z_unit : str
        Unit string for x-axis label (e.g., "m" or "km").
    figsize : (w,h)
        Figure size.

    Returns
    -------
    None
    """
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim != 1 or z_arr.size < 2:
        raise ValueError("z must be a 1D array with at least 2 points")

    A_arr = np.asarray(A)
    B_arr = np.asarray(B)

    if A_arr.ndim != 2 or A_arr.shape[0] != z_arr.shape[0] or A_arr.shape[1] != 4:
        raise ValueError("A must have shape (N,4) with N = len(z)")
    if B_arr.ndim != 2 or B_arr.shape[0] != z_arr.shape[0] or B_arr.shape[1] != 4:
        raise ValueError("B must have shape (N,4) with N = len(z)")

    if not np.iscomplexobj(A_arr):
        A_arr = A_arr.astype(np.complex128, copy=False)
    if not np.iscomplexobj(B_arr):
        B_arr = B_arr.astype(np.complex128, copy=False)

    P_A = np.sum(np.abs(A_arr) ** 2, axis=1)  # (N,)
    P_B = np.sum(np.abs(B_arr) ** 2, axis=1)  # (N,)

    plt.figure(figsize=figsize)

    if scale == "linear":
        plt.plot(z_arr, P_A, label=r"$\sum_j |A_j|^2$")
        plt.plot(z_arr, P_B, label=r"$\sum_j |B_j|^2$")
        plt.ylabel("Total power [W]")

    elif scale == "log":
        if eps <= 0.0:
            raise ValueError("eps must be > 0 for log scale")
        PA = np.maximum(P_A, eps)
        PB = np.maximum(P_B, eps)
        plt.plot(z_arr, PA, label=r"$\sum_j |A_j|^2$")
        plt.plot(z_arr, PB, label=r"$\sum_j |B_j|^2$")
        plt.ylabel("Total power [W]")
        if log_base == 10.0:
            plt.yscale("log", base=10)
        elif log_base == np.e:
            plt.yscale("log", base=np.e)
        else:
            raise ValueError("log_base must be 10.0 or np.e for scale='log'")

    elif scale == "dbW":
        if eps <= 0.0:
            raise ValueError("eps must be > 0 for dBW scale")
        PA_db = 10.0 * np.log10(np.maximum(P_A, eps))
        PB_db = 10.0 * np.log10(np.maximum(P_B, eps))
        plt.plot(z_arr, PA_db, label=r"$10\log_{10}\sum_j |A_j|^2$")
        plt.plot(z_arr, PB_db, label=r"$10\log_{10}\sum_j |B_j|^2$")
        plt.ylabel("Total power [dBW]")

    else:
        raise ValueError("scale must be one of: 'linear', 'log', 'dbW'")

    plt.xlabel(f"z [{z_unit}]")
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


# -----------------------------
# Original public functions
# -----------------------------

def plot_abs_amplitudes(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str, str, str] = ("pump 1", "pump 2", "signal", "idler"),
    title: Optional[str] = None,
    scale: str = "linear",  # "linear" | "log" | "dbW"
    show: bool = True,
    save_path: Optional[str] = None,
    z_unit: str = "m",
) -> None:
    z_arr, A_arr = _validate_z_A(z, A)
    labels = _validate_labels(wave_labels, 4, name="wave_labels")
    abs_A = np.abs(A_arr)

    if scale == "linear":
        yscale = "linear"
    elif scale == "log":
        yscale = "log"


    _plot_series(
        z_arr,
        abs_A,
        labels,
        title=title,
        xlabel=f"z [{z_unit}]",
        ylabel="|A(z)|",
        yscale=yscale,
        log_base=10,
        show=show,
        save_path=save_path,
    )


def plot_powers(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str, str, str] = ("pump 1", "pump 2", "signal", "idler"),
    title: Optional[str] = None,
    log_scale: bool = False,
    log_base: float = 10,
    log_eps: float = 1e-30,
    show: bool = True,
    save_path: Optional[str] = None,
    z_unit: str = "m",
) -> None:
    z_arr, A_arr = _validate_z_A(z, A)
    labels = _validate_labels(wave_labels, 4, name="wave_labels")

    P = np.abs(A_arr) ** 2

    if log_scale:
        P = _apply_log_clip(P, eps=log_eps)
        yscale = "log"
    else:
        yscale = "linear"

    _plot_series(
        z_arr,
        P,
        labels,
        title=title,
        xlabel=f"z [{z_unit}]",
        ylabel="P(z) [W]",
        yscale=yscale,
        log_base=log_base,
        show=show,
        save_path=save_path,
    )


def plot_signal_and_idler(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str] = ("signal", "idler"),
    title: Optional[str] = None,
    log_scale: bool = False,
    log_base: float = 10,
    log_eps: float = 1e-30,
    show: bool = True,
    save_path: Optional[str] = None,
    z_unit: str = "m",
) -> None:
    z_arr, A_arr = _validate_z_A(z, A)
    labels = _validate_labels(wave_labels, 2, name="wave_labels")

    P = np.abs(A_arr) ** 2
    P_si = P[:, 2:4]  # signal, idler

    if log_scale:
        P_si = _apply_log_clip(P_si, eps=log_eps)
        yscale = "log"
    else:
        yscale = "linear"

    _plot_series(
        z_arr,
        P_si,
        labels,
        title=title,
        xlabel=f"z [{z_unit}]",
        ylabel="P(z) [W]",
        yscale=yscale,
        log_base=log_base,
        show=show,
        save_path=save_path,
    )


def plot_signal_and_idler_separate(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str] = ("signal", "idler"),
    title: Optional[str] = None,
    log_scale: bool = False,
    log_base: float = 10,
    log_eps: float = 1e-30,
    show: bool = True,
    save_path_signal: Optional[str] = None,
    save_path_idler: Optional[str] = None,
    z_unit: str = "m",
) -> None:
    z_arr, A_arr = _validate_z_A(z, A)
    labels = _validate_labels(wave_labels, 2, name="wave_labels")

    P = np.abs(A_arr) ** 2

    if log_scale:
        P = _apply_log_clip(P, eps=log_eps)
        yscale = "log"
    else:
        yscale = "linear"

    signal_title = None if title is None else f"{title} — {labels[0]}"
    idler_title = None if title is None else f"{title} — {labels[1]}"

    _plot_series(
        z_arr,
        P[:, 2:3],
        (labels[0],),
        title=signal_title,
        xlabel=f"z [{z_unit}]",
        ylabel="P(z) [W]",
        yscale=yscale,
        log_base=log_base,
        show=show,
        save_path=save_path_signal,
    )

    _plot_series(
        z_arr,
        P[:, 3:4],
        (labels[1],),
        title=idler_title,
        xlabel=f"z [{z_unit}]",
        ylabel="P(z) [W]",
        yscale=yscale,
        log_base=log_base,
        show=show,
        save_path=save_path_idler,
    )


# -----------------------------
# New SBS-specific plotting helpers
# -----------------------------

def plot_fwm_sbs_powers_forward(
    z: np.ndarray,
    A: np.ndarray,
    *,
    labels: tuple[str, str, str, str] = ("A1", "A2", "A3", "A4"),
    title: Optional[str] = None,
    scale: str = "linear",   # "linear" | "log" | "dbW"
    log_base: float = 10,
    eps: float = 1e-30,
    show: bool = True,
    save_path: Optional[str] = None,
    z_unit: str = "km",
) -> None:
    z_arr, A_arr = _validate_z_Y(z, A, name="A")
    if A_arr.shape[1] != 4:
        raise ValueError("A must have shape (N,4) for plot_fwm_sbs_powers_forward")
    plot_power_matrix(
        z_arr,
        A_arr,
        labels=labels,
        title=title,
        scale=scale,
        log_base=log_base,
        eps=eps,
        show=show,
        save_path=save_path,
        z_unit=z_unit,
        ylabel_linear="P_A(z) [W]",
        ylabel_db="P_A(z) [dBW]",
    )


def plot_fwm_sbs_powers_backward(
    z: np.ndarray,
    B: np.ndarray,
    *,
    labels: tuple[str, str, str, str] = ("B1", "B2", "B3", "B4"),
    title: Optional[str] = None,
    scale: str = "linear",   # "linear" | "log" | "dbW"
    log_base: float = 10,
    eps: float = 1e-30,
    show: bool = True,
    save_path: Optional[str] = None,
    z_unit: str = "km",
) -> None:
    z_arr, B_arr = _validate_z_Y(z, B, name="B")
    if B_arr.shape[1] != 4:
        raise ValueError("B must have shape (N,4) for plot_fwm_sbs_powers_backward")
    plot_power_matrix(
        z_arr,
        B_arr,
        labels=labels,
        title=title,
        scale=scale,
        log_base=log_base,
        eps=eps,
        show=show,
        save_path=save_path,
        z_unit=z_unit,
        ylabel_linear="P_B(z) [W]",
        ylabel_db="P_B(z) [dBW]",
    )
