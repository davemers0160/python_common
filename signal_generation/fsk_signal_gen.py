
import math
import yaml

import numpy as np

def generate_fsk(data, num_bits, amplitude, sample_rate, bit_length, freq_separation):

    samples_per_bit = math.floor(sample_rate * bit_length)
    freq_offset = (freq_separation / 2.0) / sample_rate

    iq = np.empty(0)

    for idx in range(num_bits):

        if data[idx] == 1:
            tmp_iq = amplitude * (np.exp(1j * 2 * np.pi * freq_offset * np.arange(0, samples_per_bit)))

        else:
            tmp_iq = amplitude * (np.exp(1j * 2 * np.pi * -freq_offset * np.arange(0, samples_per_bit)))

        iq = np.concatenate((iq, tmp_iq))

    return iq


def read_input_params(filename):
    with open(filename, 'r') as file:
        fsk_params = yaml.safe_load(file)

    return fsk_params['data_rate'], fsk_params['num_bits'], fsk_params['frame_length'], fsk_params['amplitude'], fsk_params['sample_rate']


filename = "C:/Projects/data/FSK/fsk_test.yml"

data_rate, num_bits, frame_length, amplitude, sample_rate = read_input_params(filename)

#
bit_length = 1/data_rate

#
data_length = num_bits * bit_length

#
buffer_length = (frame_length - data_length)/2

#
freq_separation = 2 * data_rate

#
data = np.random.randint(2, size=num_bits)

#
sig_off = 0.001 * amplitude * (np.random.randn(math.floor(sample_rate * buffer_length)) + 1.j * np.random.randn(math.floor(sample_rate * buffer_length)))

iq = generate_fsk(data, num_bits, amplitude, sample_rate, bit_length, freq_separation)

full_signal = np.concatenate((sig_off, iq, sig_off))


data_flat = np.empty(2 * len(full_signal))
data_flat[0::2] = np.real(full_signal)
data_flat[1::2] = np.imag(full_signal)

file_name = "C:/Projects/data/FSK/fsk_test.sc16"
file_id = open(file_name, 'wb')

np.array(data_flat, dtype=np.int16).tofile(file_id)

