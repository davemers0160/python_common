# BPSK generation - returns a complex IQ file in the range [-1 + 0j, 1 + 0j]

import numpy as np
import math

#------------------------------------------------------------------------------
def generate_lfm(sample_rate, f_start, f_stop, signal_length):

    # calculate the number of samples in the RF signal
    num_samples = math.floor(sample_rate * signal_length)
    
    # time step
    t = (1.0 / sample_rate) * np.arange(0, num_samples)
      
    #v = 1i * 2.0 * M_PI * (f_start * idx * t + (f_stop - f_start) * 0.5 * idx * idx * t * t / signal_length)
    
    iq = np.exp(1j * 2.0 * np.pi * (f_start * t + (f_stop - f_start) * 0.5 * t * t / signal_length))
    
    return iq
    