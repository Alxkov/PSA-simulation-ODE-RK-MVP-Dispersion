import config, parameters, constants
import numpy as np

import io_fwm
import scan_mismtach
from parameters import make_model_params
from simulation import example_zero_signal, custom_seeded_signal
from plotting import plot_abs_amplitudes, plot_powers, plot_signal_and_idler, plot_signal_and_idler_separate

if __name__ == '__main__':
    print('Executing main')
    z, A = custom_seeded_signal()
    plot_powers(z, A)
    plot_signal_and_idler(z, A)
    # plot_signal_and_idler_separate(z, A, title=" ")
    # io_fwm.save_summary_csv("./summary.csv", z, A, overwrite=True)
    # scan_mismtach.scan_mismatch_seeded_signal(gain_mode="max")
    print('Executed successfully')

