"""
io_fwm.py

Input/output helpers for simulation results.

Design goals:
- Fast and reliable storage (NumPy .npz)
- Human-readable summaries (CSV)
- Optional metadata (JSON) for reproducibility

This module must not contain physics or numerical integration code.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import json
import csv
import datetime as _dt
from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _ensure_path(path: str | Path) -> Path:
    """Convert input to Path and expand user symbols."""
    p = Path(path).expanduser()
    return p


def _json_default(obj: Any) -> Any:
    """
    JSON serializer for objects not natively supported by json.dumps.
    - dataclasses: converted via asdict()
    - numpy scalars: converted to Python scalars
    - numpy arrays: converted to lists (careful with large arrays)
    - Path: string
    """
    if is_dataclass(obj):
        return asdict(obj)

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, np.ndarray):
        # Avoid dumping huge arrays into metadata by default:
        # if you really need arrays in metadata, store them separately in NPZ.
        return obj.tolist()

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_metadata(
    metadata: dict[str, Any] | None,
    *,
    add_timestamp: bool = True,
) -> dict[str, Any]:
    """Create a metadata dict with optional timestamp."""
    md: dict[str, Any] = {}
    if metadata:
        md.update(metadata)

    if add_timestamp and "timestamp_utc" not in md:
        md["timestamp_utc"] = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    return md


# ---------------------------------------------------------------------
# NPZ: main storage format
# ---------------------------------------------------------------------

def save_result_npz(
    path: str | Path,
    z: np.ndarray,
    A: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Save simulation result to a compressed .npz file.

    Parameters
    ----------
    path : str | Path
        Output file path. If suffix is not '.npz', it will be appended.
    z : np.ndarray
        z-grid values, shape (N,).
    A : np.ndarray
        Complex amplitudes, shape (N, 4) for your current model.
    metadata : dict | None
        Optional JSON-serializable metadata (config, params, notes, etc.).
    overwrite : bool
        If False and file exists, raises FileExistsError.

    Returns
    -------
    Path
        Path to the saved file.
    """
    p = _ensure_path(path)
    if p.suffix.lower() != ".npz":
        p = p.with_suffix(".npz")

    if p.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {p}")

    z = np.asarray(z, dtype=float)
    A = np.asarray(A)

    if z.ndim != 1:
        raise ValueError("z must be a 1D array")

    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if A.shape[0] != z.shape[0]:
        raise ValueError("A.shape[0] must match z.shape[0]")

    # Store metadata as a JSON string (keeps NPZ self-contained)
    md = _make_metadata(metadata)
    md_json = json.dumps(md, ensure_ascii=False, default=_json_default)

    p.parent.mkdir(parents=True, exist_ok=True)

    # Use compressed format: good trade-off for large arrays
    np.savez_compressed(
        p,
        z=z,
        A=A,
        metadata_json=np.array(md_json),
    )

    return p


def load_result_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load simulation result from a .npz file.

    Returns
    -------
    z : np.ndarray
        z-grid values.
    A : np.ndarray
        Complex amplitudes array.
    metadata : dict
        Metadata dict (empty if missing/unreadable).
    """
    p = _ensure_path(path)

    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    with np.load(p, allow_pickle=False) as data:
        if "z" not in data or "A" not in data:
            raise ValueError("NPZ file does not contain required keys: 'z' and 'A'")

        z = np.array(data["z"], dtype=float)
        A = np.array(data["A"])

        metadata: dict[str, Any] = {}
        if "metadata_json" in data:
            try:
                md_json = str(data["metadata_json"])
                metadata = json.loads(md_json) if md_json else {}
            except Exception:
                metadata = {}

    return z, A, metadata


# ---------------------------------------------------------------------
# JSON: metadata-only storage (optional)
# ---------------------------------------------------------------------

def save_metadata_json(
    path: str | Path,
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """
    Save metadata to a JSON file (human-readable and versionable).
    """
    p = _ensure_path(path)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")

    if p.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {p}")

    md = _make_metadata(metadata)

    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(md, f, ensure_ascii=False, indent=2, default=_json_default)

    return p


def load_metadata_json(path: str | Path) -> dict[str, Any]:
    """
    Load metadata from a JSON file.
    """
    p = _ensure_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# CSV summary: quick human-readable diagnostics
# ---------------------------------------------------------------------

def save_summary_csv(
    path: str | Path,
    z: np.ndarray,
    A: np.ndarray,
    *,
    wave_labels: tuple[str, str, str, str] = ("pump 1", "pump 2", "signal", "idler"),
    overwrite: bool = False,
) -> Path:
    """
    Save a compact CSV summary with powers and phases at each stored z.

    The file contains:
      z,
      P_pump1..P_idler,
      phi_pump1..phi_idler

    Parameters
    ----------
    path : str | Path
        Output CSV file.
    z : np.ndarray
        z values, shape (N,).
    A : np.ndarray
        Complex amplitudes, shape (N, 4).
    wave_labels : tuple[str, str, str, str]
        Column label names for the four waves.
    overwrite : bool
        If False and file exists, raises FileExistsError.

    Returns
    -------
    Path
        Path to saved CSV.
    """
    p = _ensure_path(path)
    if p.suffix.lower() != ".csv":
        p = p.with_suffix(".csv")

    if p.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {p}")

    z = np.asarray(z, dtype=float)
    A = np.asarray(A)

    if z.ndim != 1:
        raise ValueError("z must be a 1D array")

    if A.ndim != 2 or A.shape[1] != 4:
        raise ValueError("A must have shape (N, 4) for this summary function")

    if A.shape[0] != z.shape[0]:
        raise ValueError("A.shape[0] must match z.shape[0]")

    if len(wave_labels) != 4:
        raise ValueError("wave_labels must have length 4")

    # Power and phase
    P = np.abs(A) ** 2
    phi = np.angle(A)

    headers = ["z"]
    headers += [f"P_{lbl}" for lbl in wave_labels]
    headers += [f"phi_{lbl}" for lbl in wave_labels]

    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(z.shape[0]):
            row = [float(z[i])]
            row += [float(P[i, j]) for j in range(4)]
            row += [float(phi[i, j]) for j in range(4)]
            writer.writerow(row)

    return p


# ---------------------------------------------------------------------
# Convenience: save everything in one call
# ---------------------------------------------------------------------

def save_run_bundle(
    output_dir: str | Path,
    run_name: str,
    z: np.ndarray,
    A: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Save a "bundle" of outputs:
    - <run_name>.npz  (z, A, metadata_json)
    - <run_name>.csv  (powers and phases)
    - <run_name>.json (metadata only)

    Returns a dict of saved file paths.
    """
    out_dir = _ensure_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / f"{run_name}.npz"
    csv_path = out_dir / f"{run_name}.csv"
    json_path = out_dir / f"{run_name}.json"

    md = _make_metadata(metadata)

    saved: dict[str, Path] = {}
    saved["npz"] = save_result_npz(npz_path, z, A, metadata=md, overwrite=overwrite)
    saved["csv"] = save_summary_csv(csv_path, z, A, overwrite=overwrite)
    saved["json"] = save_metadata_json(json_path, md, overwrite=overwrite)

    return saved
