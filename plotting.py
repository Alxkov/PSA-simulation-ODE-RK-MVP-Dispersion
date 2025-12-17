"""
plotting.py

Visualization utilities for simulation results.

This module contains ONLY plotting code.
No physics, no numerical integration, no file I/O.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_abs_amplitudes(
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str, str, str] = (
        "pump 1",
        "pump 2",
        "signal",
        "idler",
    ),
    title: str | None = None,
    show: bool = True,
    save_path: str | None = None,
) -> None:
    """
    Plot |A_j(z)| for all four waves.

    Parameters
    ----------
    z : np.ndarray
        z-coordinates, shape (N,).
    A : np.ndarray
        Complex amplitudes, shape (N, 4).
    wave_labels : tuple of str, optional
        Labels for the four waves (legend entries).
    title : str or None, optional
        Figure title.
    show : bool, optional
        Whether to call plt.show() (default: True).
    save_path : str or None, optional
        If provided, save the figure to this path.
    """

    z = np.asarray(z, dtype=float)
    A = np.asarray(A)

    if z.ndim != 1:
        raise ValueError("z must be a 1D array")

    if A.ndim != 2 or A.shape[1] != 4:
        raise ValueError("A must have shape (N, 4)")

    if A.shape[0] != z.shape[0]:
        raise ValueError("A.shape[0] must match z.shape[0]")

    if len(wave_labels) != 4:
        raise ValueError("wave_labels must have length 4")

    # Absolute values of complex amplitudes
    abs_A = np.abs(A)

    plt.figure(figsize=(8, 5))

    for j in range(4):
        plt.plot(
            z,
            abs_A[:, j],
            label=wave_labels[j],
        )

    plt.xlabel("z [m]")
    plt.ylabel("|A(z)|")
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

def plot_powers(
        z: np.ndarray,
        A: np.ndarray,
        *,
        wave_labels: tuple[str, str, str, str] = (
                "pump 1",
                "pump 2",
                "signal",
                "idler",
        ),
        title: str | None = None,
        log_scale: bool = False,
        log_base: int = 10,
        show: bool = True,
        save_path: str | None = None,
) -> None:
    """
    Plot |A_j(z)| for all four waves.

    Parameters
    ----------
    z : np.ndarray
        z-coordinates, shape (N,).
    A : np.ndarray
        Complex amplitudes, shape (N, 4).
    wave_labels : tuple of str, optional
        Labels for the four waves (legend entries).
    title : str or None, optional
        Figure title.
    log_scale : bool, optional
        If True, use logarithmic scale on the y-axis.
    log_base : int, optional
        Logarithm base for y-axis (10 or e). Ignored if log_scale=False.
    show : bool, optional
        Whether to call plt.show().
    save_path : str or None, optional
        If provided, save the figure to this path.
    """

    z = np.asarray(z, dtype=float)
    A = np.asarray(A)

    if z.ndim != 1:
        raise ValueError("z must be a 1D array")

    if A.ndim != 2 or A.shape[1] != 4:
        raise ValueError("A must have shape (N, 4)")

    if A.shape[0] != z.shape[0]:
        raise ValueError("A.shape[0] must match z.shape[0]")

    if len(wave_labels) != 4:
        raise ValueError("wave_labels must have length 4")

    # Absolute values of complex amplitudes
    P = abs(np.conj(A)*A)

    if log_scale:
        # Log scale cannot handle non-positive values
        if np.any(P <= 0.0):
            raise ValueError(
                "Logarithmic scale requested, but P contains zero or negative values"
            )


    plt.figure(figsize=(8, 5))

    for j in range(4):
        plt.plot(
            z,
            P[:, j],
            label=wave_labels[j],
        )

    plt.xlabel("z [km]")
    plt.ylabel("P(z)")
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
