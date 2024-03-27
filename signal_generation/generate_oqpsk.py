"""
O-QPSK generation - returns a complex IQ file in the range with samples along the unit circle

bits    Angle   IQ
00      5pi/4   1/2*(-sqrt(2) - j*sqrt(2))
01      3pi/4   1/2*(-sqrt(2) + j*sqrt(2))
10      7pi/4   1/2*(sqrt(2) - j*sqrt(2))
11       pi/4   1/2*(sqrt(2) + j*sqrt(2))

In Offset QPSK one channel is offset by half a bit length

Inputs:
data: the data to modulate, in the form of a numpy array with values of 0/1

sample_rate: the sampling rate in Hz

half_bit_length: the length of time for half a bit to be transmitted in seconds

"""

import numpy as np
import math

#------------------------------------------------------------------------------
def generate_oqpsk(data, sample_rate, half_bit_length):

    samples_per_bit = math.floor(sample_rate * half_bit_length)

    # check for odd number and append a 0 at the end if it is odd
    if(data.shape[0] % 2 == 1):
        data = np.append(data, 0)

    d2 = np.reshape(data, (-1, 2))
    num_bit_pairs = d2.shape[0]
    
    # this will expand the bit to fill the right number of samples
    s = np.ones([int(2 * samples_per_bit)])
    
    v = np.sqrt(2)/2
      
    #iq = np.empty(0)
    
    # start with I and Q offset by half a bit length
    I = np.empty(0)
    Q = np.zeros([int(samples_per_bit)])
    
    for idx in range(num_bit_pairs):
    
        num = d2[idx][0]*2 + d2[idx][1]
        
        if num == 0:
            q_i = -v
            q_q = -v
        elif num == 1:
            q_i = -v
            q_q = v
        elif num == 2:
            q_i = v
            q_q = -v
        elif num == 3:
            q_i = v
            q_q = v

        # append the new data
        I = np.append(I, q_i*s)
        Q = np.append(Q, q_q*s)
        
    # add half a bit length of zeros to the I channel
    I = np.append(I, np.zeros([int(samples_per_bit)]))

    #iq = np.append(iq, (I * s) + (1j * Q * s))
    
    iq = I + 1j*Q
    
    return iq
    