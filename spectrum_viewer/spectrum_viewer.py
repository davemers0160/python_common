import platform
import os

# numpy
import numpy as np

# from scipy import signal

# File dialog stuff
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication

# import pandas as pd
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Spinner, HoverTool, Button, Div, Slider, LinearColorMapper, ColorBar
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, Spacer
# from bokeh.transform import dodge, factor_cmap, transform

# set up some global variables that will be used throughout the code
script_path = os.path.realpath(__file__)
iq_filename = ""
iq_data_path = os.path.dirname(os.path.dirname(script_path))

iq_data = []
spectrogram_data = []
f = []
t = []

spin_width = 110

app = QApplication([""])


# interactive control definitions
fft_length = Spinner(title="FFT Length", low=2, high=2**32, step=2, value=2048, width=spin_width)
fft_overlap = Spinner(title="FFT Overlap", low=0, high=2**32, step=1, value=1024, width=spin_width)
sample_rate = Spinner(title="Sample Rate (MHz)", low=0, high=2000, step=0.1, value=10, width=spin_width)
max_amp = Spinner(title="Max (dBm)", low=-200, high=100, step=0.1, value=0, width=spin_width)
min_amp = Spinner(title="Min (dBm)", low=-200, high=100, step=0.1, value=-120, width=spin_width)
adc_bits = Spinner(title="ADC Bits", low=-0, high=32, step=1, value=12, width=spin_width)
# window_size = Spinner(title="Window Sie", low=0, high=2e9, step=1, value=2048, width=spin_width)
# stop_time = Spinner(title="Stop Time (s)", low=0, high=2e9, step=0.000001, value=1, width=spin_width)

# source for the spectrogram
spectrogram_source = ColumnDataSource(data=dict(spectrogram_img=[], f=[], t=[], x=[], y=[], dw=[], dh=[]))

hover = HoverTool(tooltips=[('Freq (MHz)', '@f'), ('Time (s)', '@t'), ('Amplitude (dBm)', '@spectrogram_img')])
hover.point_policy = 'snap_to_data'
hover.line_policy = 'nearest'  #'prev'

# -----------------------------------------------------------------------------
def jet_clamp(v):
    v[v < 0] = 0
    v[v > 1] = 1
    return v

def jet_colormap(n):
    t_max = n+100
    t_min = 100

    t_range = t_max - 0
    t_avg = (t_max + 0) / 2.0
    t_m = (t_max - t_avg) / 2.0

    t = np.arange(0, t_max)

    rgb = np.empty((t_max, 3), dtype=np.uint8)
    rgb[:, 0] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg - t_m)))).astype(np.uint8)
    rgb[:, 1] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg)))).astype(np.uint8)
    rgb[:, 2] = (255*jet_clamp(1.5 - abs((4 / t_range)*(t - t_avg + t_m)))).astype(np.uint8)

    cm = ['#000000']
    for z in rgb:
        cm.append(("#" + ("{:0>2x}" * len(z))).format(*z))

    return cm

# -----------------------------------------------------------------------------
def jet_color(t, t_min, t_max):

    t_range = t_max - t_min

    p1 = t_min + t_range * (1 / 4)
    p2 = t_min + t_range * (2 / 4)
    p3 = t_min + t_range * (3 / 4)

    rgba = np.empty((t.shape[0], t.shape[1], 4), dtype=np.uint8)

    rgba[:, :, 0] = (255*jet_clamp((1.0 / (p3 - p2)) * (t - p2))).astype(np.uint8)
    rgba[:, :, 1] = (255*jet_clamp(2.0 - (1.0 / (p1 - t_min)) * abs(t - p2))).astype(np.uint8)
    rgba[:, :, 2] = (255*jet_clamp((1.0 / (p1 - p2)) * (t - p2))).astype(np.uint8)

    rgba[:, :, 3] = np.full((t.shape[0], t.shape[1]), 255, dtype=np.uint8)

    return rgba


def rgb2hex(data):
    data = np.maximum(0, np.minimum(data, 255))

    cm = []

    for idx in range(data.shape[0]):
        # print("#{0:02X}{1:02x}{2:02x}".format(data[idx, 0], data[idx, 1], data[idx, 2]))
        cm.append("#{0:02X}{1:02x}{2:02x}".format(data[idx, 0], data[idx, 1], data[idx, 2]))

    return cm

# -----------------------------------------------------------------------------
def generate_spectrogram(iq_data, N, O, fs):

    spectrogram_data = []
    win = np.hamming(N)
    quit_loop = False

    if O >= N:
        O = N//2

    for k in range(0, iq_data.shape[0] + 1, N-O):
        iq = iq_data[k:(k + N)]
        if (iq.size < N):
            iq = np.pad(iq, (0, N-iq.size), 'constant')
            quit_loop = True

        windowed_iq = iq * win
        x = np.fft.fftshift(np.fft.fft(windowed_iq, n=N))/(N*100)

        Pxx = 20 * np.log10(np.abs(x)) + 10
        # Pxx = 20 * np.log10(np.abs(x*np.conj(x)))
        spectrogram_data.append(Pxx)

        if (quit_loop == True):
            break

    spectrogram_data = np.array(spectrogram_data)
    spectrogram_data = np.nan_to_num(spectrogram_data, neginf=-200, posinf=200)

    # Frequencies:
    f = np.arange(fs/-2.0, fs/2.0, fs/N)/1e6

    # Time Range:
    dt = (N-O)/fs
    # t = np.linspace(0, iq_data.shape[0] / fs, spectrogram_data.shape[0])
    t = np.arange(dt, iq_data.shape[0] / fs, dt)
    return spectrogram_data, f, t


def update_plot(attr, old, new):
    global iq_data, spectrogram_data

    spectrogram_data, freq, time = generate_spectrogram(iq_data, fft_length.value, fft_overlap.value, sample_rate.value*1e6)

    freq = np.tile(freq, spectrogram_data.shape[0])
    time = np.repeat(time, spectrogram_data.shape[1])

    spectrogram_source.data = {'spectrogram_img': [spectrogram_data], 'x': [np.min(freq)], 'y': [0],
                               'dw': [sample_rate.value], 'dh': [np.max(time)], 't': [time], 'f': [freq]}

    bp = 1


# -----------------------------------------------------------------------------
def update_color_scale(attr, old, new):
    global spectrogram_img

    spectrogram_img.glyph.color_mapper.low = min_amp.value
    spectrogram_img.glyph.color_mapper.high = max_amp.value


def load_iq_data(attr, old, new):
    global iq_filename, iq_data_path, iq_data
    print("Processing File: ", iq_filename[0])

    bits = 2**(adc_bits.value-1)

    # load in data
    iq_data_path = os.path.dirname(iq_filename[0])

    if adc_bits.value > 8:
        x = np.fromfile(iq_filename[0], dtype=np.int16, count=-1, sep='', offset=0).astype(np.float32) / bits
    else:
        x = np.fromfile(iq_filename[0], dtype=np.int8, count=-1, sep='', offset=0).astype(np.float32) / bits

    # convert x into a complex numpy array
    x = x.reshape(-1, 2)

    iq_data = np.empty(x.shape[0], dtype=complex)
    iq_data.real = x[:, 0]
    iq_data.imag = x[:, 1]

    update_plot(1, 1, 1)

# -----------------------------------------------------------------------------
def get_input():
    global iq_filename, iq_data_path, iq_data

    iq_filename = QFileDialog.getOpenFileName(None, "Select a file",  iq_data_path, "IQ files (*.bin *.dat);;All files (*.*)")
    # iq_filename = ["D:/Projects/rf_zsl/data/sdr_test_10M_100m_0000.bin"]

    filename_div.text = "File name: " + iq_filename[0]
    if(iq_filename[0] == ""):
        return

    load_iq_data(1, 1, 1)


# -----------------------------------------------------------------------------
# the main entry point into the code
file_select_btn = Button(label='Select File', width=100)
file_select_btn.on_click(get_input)
filename_div = Div(width=800, text="File name: ", style={'font-size': '120%', 'font-weight': 'bold'})

jet_1k = jet_colormap(200)

jet_mapper = LinearColorMapper(palette=jet_colormap(200), low=min_amp.value, high=max_amp.value)
color_bar = ColorBar(color_mapper=jet_mapper, label_standoff=6)

get_input()

# define the main plot
spectrogram_fig = figure(plot_height=800, plot_width=1300, title="Spectrogram",
                         toolbar_location="right",
                         tooltips=[('Freq (MHz)', '@f{0.0000}'), ('Time (s)', '@t{0.000000}'), ('Amplitude (dBm)', '@spectrogram_img{0.0}')],
                         tools="save, pan, box_zoom, reset, wheel_zoom, hover, crosshair", active_drag="box_zoom",
                         active_scroll="wheel_zoom", active_inspect=None)

spectrogram_img = spectrogram_fig.image(image='spectrogram_img', x='x', y='y', dw='dw', dh='dh', global_alpha=1.0, dilate=False, color_mapper=jet_mapper, source=spectrogram_source)   # palette=jet_1k,

spectrogram_fig.add_layout(color_bar, 'right')
spectrogram_fig.x_range.range_padding = 0
spectrogram_fig.y_range.range_padding = 0

spectrogram_fig.title.text_font_size = "13pt"

# x-axis formatting
spectrogram_fig.xaxis.major_label_text_font_size = "12pt"
spectrogram_fig.xaxis.major_label_text_font_style = "bold"
spectrogram_fig.xaxis.axis_label_text_font_size = "14pt"
spectrogram_fig.xaxis.axis_label_text_font_style = "bold"
spectrogram_fig.xaxis.axis_label = "Frequency (MHz)"

# y-axis formatting
spectrogram_fig.yaxis.major_label_text_font_size = "12pt"
spectrogram_fig.yaxis.major_label_text_font_style = "bold"
spectrogram_fig.yaxis.axis_label_text_font_size = "14pt"
spectrogram_fig.yaxis.axis_label_text_font_style = "bold"
spectrogram_fig.yaxis.axis_label = "Time (s)"

# setup the event callbacks for the plot
for w in [fft_length, fft_overlap, sample_rate]:
    w.on_change('value', update_plot)

for w in [max_amp, min_amp]:
    w.on_change('value', update_color_scale)

for w in [adc_bits]:
    w.on_change('value', load_iq_data)

# create the layout for the controls
btn_layout = row(file_select_btn, Spacer(width=15), filename_div)
input_layout = column(fft_length, fft_overlap, sample_rate, adc_bits, max_amp, min_amp)

layout = column(btn_layout, row(input_layout, Spacer(width=20, height=20), spectrogram_fig))

# show(layout)

doc = curdoc()
doc.title = "Spectrum Viewer"
doc.add_root(layout)

