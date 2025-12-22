from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """
    general simulation configuration

    @z_max: km - max z coordinate
    @dz: km - simulation step

    @integrator: parameter step
    @save_every: number of steps after the model saved data
    @check_nan: check Nan/Inf in every step
    @verbose: verbosity
    """

    # ---- Geometry ----
    z_max: float          # [m]
    dz: float             # [m]

    # ---- Numerical method ----
    integrator: str       # 'rk4'

    # ---- Evaluation control ----
    save_every: int       # save state every save_every steps
    check_nan: bool       # check NaN / Inf each step
    verbose: bool         # print debug info


def default_simulation_config() -> SimulationConfig:
    """
    Returns
    ----------
    SimulationConfig
    """

    return SimulationConfig(
        z_max=0.5,      # km
        dz=1e-3,           # km
        integrator="rk4",
        save_every=10,
        check_nan=True,
        verbose=False,
    )

def custom_simulation_config(
        *,
        z_max=1.0,
        dz=1e-3,
        integrator="rk4",
        save_every=10,
        check_nan=True,
        verbose=False) -> SimulationConfig:
    """
    Returns
    ----------
    SimulationConfig
    """

    return SimulationConfig(
        z_max=z_max,      # km
        dz=dz,           # km
        integrator=integrator,
        save_every=save_every,
        check_nan=check_nan,
        verbose=verbose,
    )


def validate_config(cfg: SimulationConfig) -> None:
    """
    Exception
    ----------
    ValueError
        For incorrect parameters
    """
    if cfg.z_max <= 0.0:
        raise ValueError("z_max must be positive")

    if cfg.dz <= 0.0:
        raise ValueError("dz must be positive")

    if cfg.dz > cfg.z_max:
        raise ValueError("dz must be smaller than z_max")

    if cfg.integrator.lower() != "rk4":
        raise ValueError(f"Unsupported integrator: {cfg.integrator}")

    if cfg.save_every <= 0:
        raise ValueError("save_every must be a positive integer")