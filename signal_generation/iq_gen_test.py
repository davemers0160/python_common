import numpy as np
import math


sample_rate = 61.44e6

amplitude = 1950

off_time = 0.001
on_time = 0.007

sig_off = np.zeros(math.floor(sample_rate * off_time))
sig_on = amplitude * np.ones(math.floor(sample_rate * on_time))
sig_i = np.concatenate((sig_off, sig_on, sig_off))

data = sig_i.astype(np.complex)

f_offset = -17e6/sample_rate

d_r = np.exp(2 * np.pi * 1j * f_offset * np.arange(0, len(data)))

data2 = np.multiply(data, d_r)

data_flat = np.empty(2 * len(data2))
data_flat[0::2] = np.real(data2)
data_flat[1::2] = np.imag(data2)

file_name = "test.sc16"
file_id = open(file_name, 'wb')

np.array(data_flat, dtype=np.int16).tofile(file_id)

file_id.close()
bp = 1
