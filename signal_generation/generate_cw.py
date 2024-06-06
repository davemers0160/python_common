# CW generation - returns a complex IQ file in the range [-1 + 0j, 1 + 0j]

import numpy as np
import math

#------------------------------------------------------------------------------
def generate_cw(sample_rate, signal_length):

    num_samples = math.floor(sample_rate * signal_length)


    iq = np.ones([int(num_samples)]))

    # add the complex component
    iq = iq + 1j * np.zeros(iq.shape[0])
    
    return iq
    