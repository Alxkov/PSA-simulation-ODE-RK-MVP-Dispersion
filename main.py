import config, parameters, constants

if __name__ == '__main__':
    print('Executing main')
    cfg = config.SimulationConfig(
        z_max=1000, dz=0.1, integrator='rk4', save_every=10,
        check_nan=True, verbose=False)
    config.validate_config(cfg)
    params = parameters.make_parameters(cfg)
    print('Executed successfully')



