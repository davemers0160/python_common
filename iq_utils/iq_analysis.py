import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from scipy import signal

from PyQt6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, QFormLayout, QLineEdit, QDialogButtonBox

from iq_utils import read_binary_iq_data

import mplcursors

#------------------------------------------------------------------------------
class InputDialog(QDialog):
    """
    Custom input dialog to replicate MATLAB's inputdlg.
    It takes a list of prompts and default values.
    """
    def __init__(self, prompts, defaults, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Input')
        
        self.layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        
        self.line_edits = []
        for i, prompt_text in enumerate(prompts):
            line_edit = QLineEdit(self)
            if i < len(defaults):
                line_edit.setText(defaults[i])
            form_layout.addRow(prompt_text, line_edit)
            self.line_edits.append(line_edit)
            
        self.layout.addLayout(form_layout)
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        self.layout.addWidget(buttons)

    # ------------------------------------------------------------------------------
    def get_values(self):
        """Returns the text from all line edits as a list."""
        return [le.text() for le in self.line_edits]


# ------------------------------------------------------------------------------
def main():
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
    file_filter = 'IQ Files (*.sc16 *.fc32 *.iq *.sigmf-data);;All Files (*.*)'
    # The MATLAB code uses 'MultiSelect', 'on' but then processes a single file.
    # This implementation selects a single file for simplicity and to match the logic.
    data_file_path, _ = QFileDialog.getOpenFileName(None, 'Select File', startpath, file_filter)

    if not data_file_path:
        print("No file selected. Exiting.")
        return

    # fileparts() equivalent
    data_filepath, data_file = os.path.split(data_file_path)
    fn, ext = os.path.splitext(data_file)

    scale = 1.0
    data_type = ''

    # strcmp(ext, '.fc32') equivalent
    if ext == '.fc32':
        scale = 32768.0 / 2048.0
        data_type = 'single'
    elif ext == '.sc16':  # Corrected from .sc26 in original code
        scale = 1.0 / 2048.0
        data_type = 'int16'
    elif ext == '.sigmf-data':
        scale = 1.0 / 2048.0
        data_type = 'int16'
    elif ext == '.iq':
        scale = 1.0 / 2048.0
        data_type = 'int16'

    byte_order = 'little'

    ## assumes that the data is int16 and little endian
    iqc_in = read_binary_iq_data(data_file_path)
    if iqc_in is None:
        print("Failed to read IQ data.")
        return

    # get sample rate
    # prompts = ['Sample Rate:', 'Down Samples Rate:', 'Num Taps:']
    # definput = ['20e6', '1', '1']
    prompts = ['Sample Rate:']
    definput = ['20e6']

    dialog = InputDialog(prompts, definput)

    # if(isempty(res)) -> return;
    if not dialog.exec():
        print("User cancelled the input dialog. Exiting.")
        return

    res = dialog.get_values()

    # str2double() equivalent
    fs_o = np.float64(res[0])
    # ds_rate = np.float64(res[1])
    # num_taps = np.float64(res[2])

    # t = 0:1/fs_o:(numel(iqc_in)-1)/fs_o; # This line was commented out

    # %%
    # fprintf() and max/min/real/imag equivalents
    print(f'max real: {np.max(iqc_in.real)}')
    print(f'min real: {np.min(iqc_in.real)}')
    print(f'max imag: {np.max(iqc_in.imag)}')
    print(f'min imag: {np.min(iqc_in.imag)}')

    # max_v = max([...]) can be simplified in numpy
    max_v = np.max(np.abs(iqc_in))
    print(f'overall max: {max_v}')

    iqc = scale * iqc_in
    fs = fs_o

    #------------------------------------------------------------------------------
    # FFT
    Y = np.fft.fft(iqc) / len(iqc)
    f = np.fft.fftshift(np.fft.fftfreq(len(Y), d=1 / fs))
    plt.figure(num=plot_num, figsize=(14, 5), facecolor='w')
    plt.plot(f * 1e-6, 20 * np.log10(np.abs(np.fft.fftshift(Y))), 'b')
    plt.grid(True)
    plt.xlabel('Frequency (MHz)', fontweight='bold')
    plt.ylabel('Amplitude (dB)', fontweight='bold')
    plt.show(block=False)
    plot_num += 1

    #------------------------------------------------------------------------------
    # Time Domain Plot
    t = np.arange(len(iqc)) / fs
    plt.figure(num=plot_num, figsize=(14, 5), facecolor='w')
    plt.plot(t, iqc.real, 'b', label='Real (I)')
    plt.plot(t, iqc.imag, 'r', label='Imaginary (Q)')
    plt.grid(True)
    plt.xlabel('Time (s)', fontweight='bold')
    plt.ylabel('Amplitude', fontweight='bold')
    plt.ylim([-1.2, 1.2])
    plt.legend()
    plt.show(block=False)
    plot_num += 1

    #------------------------------------------------------------------------------
    # Spectrogram
    f_spec, ts_spec, Sxx = signal.spectrogram(iqc, fs, nperseg=512, noverlap=256, nfft=512, return_onesided=False)
    Sxx_shifted = np.fft.fftshift(Sxx+1.0e-8, axes=0)
    f_spec_shifted = np.fft.fftshift(f_spec)
    Sxx_db = 20 * np.log10(np.abs(Sxx_shifted))
    Sxx_db = np.maximum(Sxx_db, -110)  # Set floor to -110 dB

    plt.figure(num=plot_num, figsize=(10, 8), facecolor='w')
    plt.pcolormesh(ts_spec, f_spec_shifted / 1e6, Sxx_db, shading='gouraud', cmap='jet')
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('Time (s)', fontweight='bold', fontsize=12)
    plt.ylabel('Frequency (MHz)', fontweight='bold', fontsize=12)
    plt.gca().tick_params(axis='both', which='major', labelsize=12)
    plt.show(block=False)
    plot_num += 1

    #------------------------------------------------------------------------------
    # Constellation Plot
    plt.figure(num=plot_num, figsize=(8, 5), facecolor='w')
    ax_const = plt.gca()
    ax_const.scatter(iqc.real, iqc.imag, marker='o', c='b', s=5)  # s is marker size
    ax_const.grid(True)
    ax_const.set_xlim([-1.2, 1.2])
    ax_const.set_ylim([-1.2, 1.2])
    ax_const.set_xlabel('I', fontweight='bold', fontsize=12)
    ax_const.set_ylabel('Q', fontweight='bold', fontsize=12)
    # Move axes to origin
    ax_const.spines['left'].set_position('zero')
    ax_const.spines['bottom'].set_position('zero')
    ax_const.spines['right'].set_color('none')
    ax_const.spines['top'].set_color('none')
    plot_num += 1

    #------------------------------------------------------------------------------
    # 3D Time-IQ Plot
    iq_start = max(int(fs * 0.000001), 0)
    iq_stop = min(iq_start + int(np.ceil(fs * 0.01)), len(iqc))  # shorter duration for clarity
    step = 10  # step to avoid overplotting
    fig3d = plt.figure(num=plot_num, figsize=(14, 5), facecolor='w')
    ax3d = fig3d.add_subplot(111, projection='3d')
    t_slice = t[iq_start:iq_stop:step]
    iq_slice = iqc[iq_start:iq_stop:step]
    ax3d.scatter(t_slice, iq_slice.real, iq_slice.imag, s=20, c='b', marker='o')
    ax3d.plot(t_slice, iq_slice.real, iq_slice.imag, 'b')
    ax3d.invert_yaxis()  # Equivalent to 'Ydir','reverse'
    ax3d.set_xlabel('Time (s)', fontweight='bold')
    ax3d.set_ylabel('I', fontweight='bold')
    ax3d.set_zlabel('Q', fontweight='bold')
    plt.show(block=False)
    plot_num += 1

    #------------------------------------------------------------------------------
    # Constellation Histogram
    iq_start = 0 #max(0, int(np.ceil(fs * 0.0001)))
    iq_stop = min(iq_start + int(np.ceil(fs * 0.001)), len(iqc))
    iq_section = iqc[iq_start:iq_stop]
    x_edges = np.arange(-1, 1.05, 0.05)
    y_edges = np.arange(-1, 1.05, 0.05)

    iq_hist, xedges, yedges = np.histogram2d(iq_section.real, iq_section.imag, bins=(x_edges, y_edges))
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    fig_hist = plt.figure(facecolor='w')
    ax_hist = fig_hist.add_subplot(111, projection='3d')
    ax_hist.plot_surface(X, Y, iq_hist.T, cmap='jet', edgecolor='none')  # Transpose hist matrix
    plt.title('Constellation Histogram')
    plt.xlabel('I')
    plt.ylabel('Q')
    plot_num += 1


    # Create a new figure and axes for the 2D plot
    plt.figure(f"Constellation Heatmap ({plot_num})", figsize=(9, 7), facecolor='w')
    ax_hist = plt.gca()

    # Use pcolormesh to create the heatmap. Note the transpose on iq_hist.
    im = ax_hist.pcolormesh(xedges, yedges, iq_hist.T, cmap='jet', shading='auto')

    # Add a color bar to show the scale
    plt.colorbar(im, ax=ax_hist, label='Number of Points')

    # Set labels and title
    ax_hist.set_title('Constellation Heatmap', fontweight='bold')
    ax_hist.set_xlabel('I Bins', fontweight='bold')
    ax_hist.set_ylabel('Q Bins', fontweight='bold')
    ax_hist.set_aspect('equal', adjustable='box')  # Keep the aspect ratio square
    plot_num += 1

    # Enable the data cursors on all plots. hover=True is recommended.
    mplcursors.cursor(multiple=True)

    # plt.show()
    plt.show(block=True)

    # input("Press Enter to close all plots and end the program...")

if __name__ == '__main__':
    main()