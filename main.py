import config, parameters, constants
import numpy as np

from parameters import make_model_params
from simulation import example_zero_signal

if __name__ == '__main__':
    print('Executing main')
    z, A = example_zero_signal()
    print(z[-1], A[-1])
    print('Executed successfully')



