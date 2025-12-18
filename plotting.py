"""
plotting.py

Visualization utilities for simulation results.

This module contains ONLY plotting code.
No physics, no numerical integration, no file I/O.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Internal validation helpers
# -----------------------------

def _validate_z_A(z: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_arr = np.asarray(z, dtype=float)
    A_arr = np.asarray(A)

    if z_arr.ndim != 1:
        raise ValueError("z must be a 1D array")

    if A_arr.ndim != 2 or A_arr.shape[1] != 4:
        raise ValueError("A must have shape (N, 4)")

    if A_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("A.shape[0] must match z.shape[0]")

    return z_arr, A_arr


def _validate_labels(labels: Sequence[str], n: int, *, name: str = "wave_labels") -> tuple[str, ...]:
    if len(labels) != n:
        raise ValueError(f"{name} must have length {n}")
    return tuple(labels)


def _apply_log_scale(y: np.ndarray, *, log_base: float, eps: float) -> np.ndarray:
    """
    Prepare y for log-scale plotting.

    We clip values to at least eps to avoid log(0).
    """
    if eps <= 0.0:
        raise ValueError("eps must be > 0 for log-scale clipping")

    y_safe = np.maximum(y, eps)
    # Note: we don't take logarithm ourselves; matplotlib does that.
    # We only ensure positivity.
    return y_safe


# -----------------------------
# One plotting "engine"
# -----------------------------

def _plot_series(
    z: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    *,
    title: str | None,
    xlabel: str,
    ylabel: str,
    log_scale: bool,
    log_base: float,
    log_eps: float,
    show: bool,
    save_path: str | None,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> None:
    z_arr = np.asarray(z, dtype=float)
    y_arr = np.asarray(y)

    if z_arr.ndim != 1:
        raise ValueError("z must be a 1D array")
    if y_arr.ndim != 2:
        raise ValueError("y must be a 2D array with shape (N, M)")
    if y_arr.shape[0] != z_arr.shape[0]:
        raise ValueError("y.shape[0] must match z.shape[0]")
    if y_arr.shape[1] != len(labels):
        raise ValueError("labels length must match y.shape[1]")

    if log_scale:
        y_arr = _apply_log_scale(y_arr, log_base=log_base, eps=log_eps)

    plt.figure(figsize=figsize)

    for j, lab in enumerate(labels):
        plt.plot(z_arr, y_arr[:, j], label=lab)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if log_scale:
        if log_base == 10:
            plt.yscale("log", base=10)
        elif log_base == np.e:
            plt.yscale("log", base=np.e)
        else:
            raise ValueError("log_base must be 10 or np.e")

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
# Public functions (thin wrappers)
# -----------------------------

def plot_abs_amplitudes(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str, str, str] = ("pump 1", "pump 2", "signal", "idler"),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    z_arr, A_arr = _validate_z_A(z, A)
    labels = _validate_labels(wave_labels, 4)

    abs_A = np.abs(A_arr)

    _plot_series(
        z_arr,
        abs_A,
        labels,
        title=title,
        xlabel="z [m]",
        ylabel="|A(z)|",
        log_scale=False,
        log_base=10,
        log_eps=1e-30,
        show=show,
        save_path=save_path,
    )


def plot_powers(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str, str, str] = ("pump 1", "pump 2", "signal", "idler"),
    title: str | None = None,
    log_scale: bool = False,
    log_base: float = 10,
    log_eps: float = 1e-30,
    show: bool = True,
    save_path: str | None = None,
    z_unit: str = "m",
) -> None:
    z_arr, A_arr = _validate_z_A(z, A)
    labels = _validate_labels(wave_labels, 4)

    P = np.abs(A_arr) ** 2

    xlabel = f"z [{z_unit}]"
    _plot_series(
        z_arr,
        P,
        labels,
        title=title,
        xlabel=xlabel,
        ylabel="P(z) [W]",
        log_scale=log_scale,
        log_base=log_base,
        log_eps=log_eps,
        show=show,
        save_path=save_path,
    )


def plot_signal_and_idler(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str] = ("signal", "idler"),
    title: str | None = None,
    log_scale: bool = False,
    log_base: float = 10,
    log_eps: float = 1e-30,
    show: bool = True,
    save_path: str | None = None,
    z_unit: str = "m",
) -> None:
    z_arr, A_arr = _validate_z_A(z, A)
    labels = _validate_labels(wave_labels, 2)

    P = np.abs(A_arr) ** 2
    P_si = P[:, 2:4]  # columns: signal, idler

    xlabel = f"z [{z_unit}]"
    _plot_series(
        z_arr,
        P_si,
        labels,
        title=title,
        xlabel=xlabel,
        ylabel="P(z) [W]",
        log_scale=log_scale,
        log_base=log_base,
        log_eps=log_eps,
        show=show,
        save_path=save_path,
    )

def plot_signal_and_idler_separate(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str] = ("signal", "idler"),
    title: str | None = None,
    log_scale: bool = False,
    log_base: float = 10,
    log_eps: float = 1e-30,
    show: bool = True,
    save_path_signal: str | None = None,
    save_path_idler: str | None = None,
    z_unit: str = "m",
) -> None:
    """
    Plot signal and idler powers as two separate figures.

    Parameters
    ----------
    z : np.ndarray
        z-coordinates, shape (N,).
    A : np.ndarray
        Complex amplitudes, shape (N, 4) in order (pump1, pump2, signal, idler).
    wave_labels : tuple[str, str], optional
        Labels for (signal, idler).
    title : str or None, optional
        Base title for figures. If provided, titles become "<title> — signal" and "<title> — idler".
    log_scale : bool, optional
        If True, use logarithmic scale on the y-axis.
    log_base : float, optional
        Logarithm base for y-axis (10 or np.e). Ignored if log_scale=False.
    log_eps : float, optional
        Lower clipping bound for log-scale plotting (prevents log(0)).
    show : bool, optional
        Whether to call plt.show() (for each figure).
    save_path_signal : str or None, optional
        If provided, save the signal figure to this path.
    save_path_idler : str or None, optional
        If provided, save the idler figure to this path.
    z_unit : str, optional
        Unit label for z-axis, e.g. "m" or "km".
    """
    z_arr, A_arr = _validate_z_A(z, A)
    labels = _validate_labels(wave_labels, 2)

    P = np.abs(A_arr) ** 2
    xlabel = f"z [{z_unit}]"

    signal_title = None if title is None else f"{title} — {labels[0]}"
    idler_title = None if title is None else f"{title} — {labels[1]}"

    # Signal (column 2)
    _plot_series(
        z_arr,
        P[:, 2:3],
        (labels[0],),
        title=signal_title,
        xlabel=xlabel,
        ylabel="P(z) [W]",
        log_scale=log_scale,
        log_base=log_base,
        log_eps=log_eps,
        show=show,
        save_path=save_path_signal,
    )

    # Idler (column 3)
    _plot_series(
        z_arr,
        P[:, 3:4],
        (labels[1],),
        title=idler_title,
        xlabel=xlabel,
        ylabel="P(z) [W]",
        log_scale=log_scale,
        log_base=log_base,
        log_eps=log_eps,
        show=show,
        save_path=save_path_idler,
    )