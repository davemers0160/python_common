
import numpy as np
import math

def maximal_length_sequence(reg_length, taps):

    register = np.zeros([reg_length])
    register[0] = 1

    sr_size = 2**reg_length - 1
    SR = np.zeros([sr_size])

    for idx in range(sr_size):
        SR[idx] = register[-1]

        tmp_sum = 0
        for jdx in range(taps.shape[0]):
            tmp_sum += register[taps[jdx]]

        register[1:] = register[0:-1]
        register[0] = tmp_sum % 2

    SR = 2*SR - 1
    return SR
