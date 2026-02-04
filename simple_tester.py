import os
import sys
import numpy as np
import tomllib

from scipy import signal
from scipy.fft import fft, fftfreq, fftshift

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

from PyQt6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox

from iq_utils.read_sos_coefficients import read_complex_sos_coefficients
from iq_utils.binary_file_ops import read_binary_iq_data

# In Python, we create a QApplication instance to handle UI elements.
# The QApplication does not need to be run with app.exec()
# because the dialogs are modal and block execution themselves.
# The app instance is garbage collected at the end of the script.
app = QApplication(sys.argv)

# format long g, format compact, clc, close all, clearvars
# These MATLAB commands control the workspace and display. In Python,
# scripts run in a clean environment, and print formatting is explicit.
# np.set_printoptions can be used for more control if needed.

plot_num = 1

# Gets the path of the current script to use as the starting directory
try:
    # This works when running as a script
    full_path = os.path.abspath(__file__)
except NameError:
    # Fallback for interactive environments like Jupyter
    full_path = os.path.abspath('.')

startpath = os.path.dirname(full_path)

# load in data
file_filter = 'CSV Files (*.csv);;TOML Files (*.toml);;IQ Files (*.sc16 *.fc32 *.iq *.sigmf-data);;All Files (*.*)'
# The MATLAB code uses 'MultiSelect', 'on' but then processes a single file.
# This implementation selects a single file for simplicity and to match the logic.
data_file_path, _ = QFileDialog.getOpenFileName(None, 'Select File', startpath, file_filter)

if not data_file_path:
    print("No file selected. Exiting.")
    exit(0)

# fileparts() equivalent
data_filepath, data_file = os.path.split(data_file_path)


# filename = "d:/Projects/data/RF/sb_test/test_coeff.csv"
# d = read_complex_sos_coefficients(data_file_path)


filter = "filter={type=\"FT_IIR\", coeff=[[\"0.017286995+0.000000000j\", \"-0.024107483+0.000000000j\", \"0.017286995+0.000000000j\", \"1.000000000+0.000000000j\", \"-1.477907107+0.000000000j\", \"0.868917979+0.000000000j\"], \
        [\"1.000000000+0.000000000j\", \"-1.204682007+0.000000000j\", \"1.000000000+0.000000000j\", \"1.000000000+0.000000000j\", \"-1.209561054+0.000000000j\", \"0.613943971+0.000000000j\"]]}"

'''
        [1.000000000+0.000000000j, -0.570914261+0.000000000j, 1.000000000+0.000000000j, 1.000000000+0.000000000j, -0.871565079+0.000000000j, 0.336175109+0.000000000j], \
        [1.000000000+0.000000000j, 1.273774378+0.000000000j, 1.000000000+0.000000000j, 1.000000000+0.000000000j, -0.576420272+0.000000000j, 0.106546618+0.000000000j], \
        [0.984353722+0.000000000j, -1.662234672+1.054886200j, 0.419117430-0.890669877j, 1.000000000+0.000000000j, -1.662130507+1.054820095j, 0.412508097-0.876624329j], \
        [0.984353722+0.000000000j, -1.725191484+0.948432044j, 0.527443100-0.831117336j, 1.000000000+0.000000000j, -1.725083374+0.948372610j, 0.519125510-0.818010911j], \
        [0.968830814+0.000000000j, -1.823108562+0.656359470j, 0.746496972-0.617556003j, 1.000000000+0.000000000j, -1.822644256+0.656192309j, 0.722860934-0.598002572j], \
        [0.984353722+0.000000000j, -1.959970250+0.185271737j, 0.966918111-0.184449494j, 1.000000000+0.000000000j, -1.959847428+0.185260127j, 0.951670156-0.181540791j], \
        [0.968830814+0.000000000j, -1.929062216-0.182350068j, 0.951670156+0.181540791j, 1.000000000+0.000000000j, -1.928570926-0.182303627j, 0.921537801+0.175792737j], \
        [0.984353722+0.000000000j, -1.872352043-0.608364057j, 0.796358889+0.578588601j, 1.000000000+0.000000000j, -1.872234711-0.608325934j, 0.783800593+0.569464464j], \
        [0.968830814+0.000000000j, -1.697985828-0.933475608j, 0.519125510+0.818010911j, 1.000000000+0.000000000j, -1.697553388-0.933237872j, 0.502688644+0.792110556j]]}"
'''

# filter = "filter={type=\"FT_IIR\", coeff=[[\"1+2j\", \"3+4j\"],[\"5-4j\", \"3-2j\"]]}"


with open(data_file_path, "rb") as f:
    data = tomllib.load(f)

print(data)

raw_coeff = data["filter"]["coeff"]

# complex_coeff = [ [complex(item.replace(" ", "")) for item in row] for row in raw_coeff ]

complex_array = np.array(raw_coeff, dtype=np.complex128)

bp = 1


filename = "D:/Projects/data/RF/sb_test/sb_iq_frame[000].sigmf-data"
data = read_binary_iq_data(filename)
num_samples = data.size

iq_filt = signal.sosfilt(d, data)
sample_rate = 10e6  # 10 MHz

fft_raw = fft(iq_filt)

# Shift the zero-frequency component to the center
fft_shifted = fftshift(fft_raw)

# Generate frequency axis and shift it to match
frequency_axis = fftshift(fftfreq(num_samples, d=1 / sample_rate))

# Calculate magnitude in dB
# Adding a small epsilon to avoid log10(0)
magnitude_db = 20 * np.log10(np.abs(fft_shifted) + 1e-12)

plt.figure(figsize=(12, 6))

# plot the magnitude spectrum
plt.plot(frequency_axis / 1e6, magnitude_db, color='tab:blue', linewidth=1)
plt.title("FFT of SOS-Filtered Signal")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.show(block=True)

bp = 2
