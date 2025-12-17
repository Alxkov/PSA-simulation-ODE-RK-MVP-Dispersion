import config, parameters, constants
import numpy as np

import io_fwm
from parameters import make_model_params
from simulation import example_zero_signal, example_seeded_signal
from plotting import plot_abs_amplitudes, plot_powers

if __name__ == '__main__':
    print('Executing main')
    z, A = example_seeded_signal()
    plot_powers(z, A)
    io_fwm.save_summary_csv("./summary.csv", z, A, overwrite=True)
    print('Executed successfully')

