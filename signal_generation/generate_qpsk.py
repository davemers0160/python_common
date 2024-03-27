"""
QPSK generation - returns a complex IQ file in the range with samples along the unit circle

bits    Angle   IQ
00      5pi/4   1/2*(-sqrt(2) - j*sqrt(2))
01      3pi/4   1/2*(-sqrt(2) + j*sqrt(2))
10      7pi/4   1/2*(sqrt(2) - j*sqrt(2))
11       pi/4   1/2*(sqrt(2) + j*sqrt(2))

"""

import numpy as np
import math

#------------------------------------------------------------------------------
def generate_qpsk(data, sample_rate, bit_length):

    samples_per_bit = math.floor(sample_rate * bit_length)

    # check for odd number and append a 0 at the end if it is odd
    if(data.shape[0] % 2 == 1):
        data = np.append(data, 0)

    d2 = np.reshape(data, (-1, 2))
    num_bit_pairs = d2.shape[0]
    
    # this will expand the bit to fill the right number of samples
    s = np.ones([int(samples_per_bit)])

    v = np.sqrt(2)/2

    iq = np.empty(0)
    for idx in range(num_bit_pairs):
    
        num = d2[idx][0]*2 + d2[idx][1]
        
        if num == 0:
            I = -v
            Q = -v
        elif num == 1:
            I = -v
            Q = v
        elif num == 2:
            I = v
            Q = -v
        elif num == 3:
            I = v
            Q = v
    
        iq = np.append(iq, (I * s) + (1j * Q * s))
    
    return iq
    