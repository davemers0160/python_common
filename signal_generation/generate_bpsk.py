# BPSK generation - returns a complex IQ file in the range [-1 + 0j, 1 + 0j]

import numpy as np
import math

#------------------------------------------------------------------------------
def generate_bpsk(data, sample_rate, bit_length):

    num_bits = data.shape[0]
    samples_per_bit = math.floor(sample_rate * bit_length)

    iq = np.empty(0)
    for idx in range(num_bits):
        iq = np.append(iq, data[idx] * np.ones([int(samples_per_bit)]))

    # add the complex component
    iq = iq + 1j * np.zeros(iq.shape[0])
    
    return iq
    