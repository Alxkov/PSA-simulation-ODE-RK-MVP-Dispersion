import config, parameters, constants
import numpy as np

from parameters import make_model_params

if __name__ == '__main__':
    print('Executing main')
    cfg = config.SimulationConfig(
        z_max=100, dz=0.1, integrator='rk4', save_every=10,
        check_nan=True, verbose=False)
    config.validate_config(cfg)
    fiber_params = parameters.make_fiber_params(
        gamma=10,
        beta=np.array([9.24, 9.24, 9.24, 9.24]),
        alpha=0)
    omega = constants.c/(1.55e-6) # rad/s
    inp_powers = [1, 1, 1e-3, 0]
    waves_params = parameters.make_wave_params(
        omega=np.array([omega, omega, omega, omega]),
        P_in=np.array([*inp_powers]),
    )
    initial_condition = parameters.make_initial_conditions(waves_params.P_in)

    params = make_model_params(fiber_params, waves_params, initial_condition)
    print(params)

    print('Executed successfully')



