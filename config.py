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

    # ---- Геометрия интегрирования ----
    z_max: float          # [m] максимальная координата z (длина моделирования)
    dz: float             # [m] шаг интегрирования

    # ---- Численный метод ----
    integrator: str       # 'rk4' (фиксированный шаг)

    # ---- Контроль расчёта ----
    save_every: int       # сохранять состояние каждые N шагов
    check_nan: bool       # проверять NaN / Inf на каждом шаге
    verbose: bool         # печатать диагностические сообщения


def default_simulation_config() -> SimulationConfig:
    """
    Конфигурация по умолчанию — безопасная для большинства FWM/OPA расчётов.

    Возвращает
    ----------
    SimulationConfig
        Объект конфигурации моделирования
    """

    return SimulationConfig(
        z_max=100.0,      # km
        dz=1.0,           # km
        integrator="rk4",
        save_every=10,
        check_nan=True,
        verbose=False,
    )


def validate_config(cfg: SimulationConfig) -> None:
    """
    Проверка корректности конфигурации.

    Параметры
    ----------
    cfg : SimulationConfig

    Исключения
    ----------
    ValueError
        Если параметры некорректны
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