
import math
import yaml

import numpy as np
from filter_windows import nuttall_window, create_fir_filter

def generate_fsk(data, num_bits, amplitude, sample_rate, bit_length, center_freq, freq_separation):

    samples_per_bit = math.floor(sample_rate * bit_length)
    # freq_offset = (freq_separation / 2.0) / sample_rate

    f1 = (center_freq - freq_separation)/sample_rate
    f2 = (center_freq + freq_separation)/sample_rate

    iq = np.empty(0)

    for idx in range(num_bits):

        if data[idx] == 1:
            tmp_iq = amplitude * (np.exp(1j * 2 * np.pi * f1 * np.arange(0, samples_per_bit)))

        else:
            tmp_iq = amplitude * (np.exp(1j * 2 * np.pi * f2 * np.arange(0, samples_per_bit)))

        iq = np.concatenate((iq, tmp_iq))

    return iq


def read_input_params(filename):
    with open(filename, 'r') as file:
        fsk_params = yaml.safe_load(file)

    return fsk_params['data_rate'], fsk_params['num_bits'], fsk_params['frame_length'], fsk_params['amplitude'], \
           fsk_params['center_frequency'], fsk_params['sample_rate']


filename = "C:/Projects/data/FSK/fsk_test.yml"

data_rate, num_bits, frame_length, amplitude, center_freq, sample_rate = read_input_params(filename)

#
bit_length = 1.0/data_rate

#
data_length = num_bits * bit_length

#
buffer_length = (frame_length - data_length)

#
freq_separation = data_rate

#
data = np.random.randint(2, size=num_bits)

#
sig_off = 0.001 * amplitude * (np.random.randn(math.floor(sample_rate * buffer_length)) + 1.j * np.random.randn(math.floor(sample_rate * buffer_length)))

iq = generate_fsk(data, num_bits, amplitude, sample_rate, bit_length, center_freq, freq_separation)

n_taps = 256
fc = (2*freq_separation)/sample_rate
w = nuttall_window(n_taps)
lpf = create_fir_filter(fc, w)

#fc_rot = exp(1.0j*2.0*pi()* f_offset/sample_rate*(0:(numel(iq_bpsk)-1))).';
f_offset = center_freq/sample_rate
bpf = lpf * np.exp(-1j * 2 * np.pi * f_offset*np.arange(n_taps))

full_signal = np.concatenate((iq, sig_off))

full_signal = np.convolve(full_signal, np.flip(bpf), 'same')



data_flat = np.empty(2 * len(full_signal))
data_flat[0::2] = np.real(full_signal)
data_flat[1::2] = np.imag(full_signal)

file_name = "C:/Projects/data/FSK/fsk_test.sc16"
file_id = open(file_name, 'wb')

np.array(data_flat, dtype=np.int16).tofile(file_id)

file_id.close()
