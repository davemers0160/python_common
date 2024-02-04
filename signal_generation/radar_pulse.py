# This is a sample Python script to generate BPSK radar pulses.

import os
import sys

import numpy as np
import yaml
import argparse

from sequences import maximal_length_sequence
from generate_bpsk import generate_bpsk
from filter_windows import nuttall_window, create_fir_filter

## do this because the relative path doesn't work on linux without PYTHONPATH
script_path = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, os.path.abspath(project_path))
from iq_utils.binary_file_ops import write_binary_iq_data

#------------------------------------------------------------------------------
def read_input_params(filename):
    with open(filename, 'r') as file:
        input_params = yaml.safe_load(file)

    return input_params['sample_rate'], input_params['bit_length'], input_params['num_bits'], input_params['taps'], \
        input_params['amplitude'], input_params['pri'], input_params['num_pulses']


#------------------------------------------------------------------------------
def generate_pulse(seq, sample_rate, bit_length, pri, num_pulses):

    samples_per_pulse = sample_rate * pri

    pulse = generate_bpsk(seq, sample_rate, bit_length)

    # add zeros to the end of the pulse until the pri is satisfied
    pulse = np.append(pulse, np.zeros([int(samples_per_pulse - pulse.shape[0])]))

    # create the filter parameters
    fc = (2.05/bit_length)/sample_rate
    num_taps = 501

    # create the filter
    w = nuttall_window(num_taps)
    lpf = create_fir_filter(fc, w)

    # filter the pulse
    pulse_filt = np.convolve(pulse, lpf[::-1], 'same')

    # normalize the pulse
    pulse_max = np.max(np.abs(pulse_filt))
    pulse_filt = pulse_filt/pulse_max

    # append multiple copies
    pulse_seq = np.tile(pulse_filt, [num_pulses])

    return pulse_seq


#------------------------------------------------------------------------------
# MAIN
# add an argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file", help="input audio file")
parser.add_argument("-o", "--output_file", help="output IQ file (*.sc16)")

# Read arguments from command line
args = parser.parse_args()

if not args.input_file or not args.output_file:
    print("No input or output file specified")
    exit(1)

# input parameters
filename = args.input_file
save_filename = args.output_file

# read in the yaml input file
sample_rate, bit_length, num_bits, sequence_taps, amplitude, pri, num_pulses = read_input_params(filename)

# generate the data sequence
seq = maximal_length_sequence(num_bits, np.array(sequence_taps))

print("Generating IQ file...")
pulse_seq = generate_pulse(seq, sample_rate, bit_length, pri, num_pulses)

# write the IQ data to a binary file
print("Saving IQ file...")
write_binary_iq_data(save_filename, np.round(amplitude * pulse_seq))

print("Complete!")
