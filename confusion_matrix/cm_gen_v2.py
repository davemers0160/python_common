import os
import math
import numpy as np

from bokeh import events
from bokeh.io import curdoc, output_file
from bokeh.models import ColumnDataSource, Button, Div, Legend, CustomJS, HoverTool, Range1d, LinearColorMapper, CategoricalColorMapper
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row, Spacer
from bokeh.palettes import Magma, Magma256, magma, viridis
from bokeh.sampledata.periodic_table import elements
from bokeh.transform import dodge, factor_cmap, transform

import pandas as pd

# File dialog stuff
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QFileDialog, QWidget, QApplication

# global variables used throughout the code
# required for QT to use the button
app = QApplication([""])
output_file("test.html")

# path to start looking in for confusion matrix files
start_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# get the min and max for the color shading of the confusion matrix.  Min is assumed to be 0.
cm_min = 0
cm_max = 100

# plot specific variables that are used in plot formatting (size/color)...
cm_plot_h = 800
cm_plot_w = 1500
err_plot_h = 130
err_plot_w = cm_plot_w

error_colors = ["#00FF00", "#FFA700", "#FF0000"]
text_mapper = LinearColorMapper(palette=["#000000", "#000000", "#FFFFFF"], low=75, high=cm_max)



# cm_data = np.array([[20, 10], [5, 15]])
cm_data = pd.read_csv("D:/Projects/dfd/dfd_dnn_analysis/results/tb23a_test/tb23a_confusion_matrix_results.txt", header=None).values

cm_data_size = cm_data.shape[0]

# dm_values = np.arange(start=0, stop=cm_data.shape[0])
dm_values_str = [str(x) for x in range(0, cm_data_size)]

cm_y, cm_x = np.indices([cm_data_size, cm_data_size])
cm_x = np.hstack(cm_x)
cm_y = np.hstack(cm_y)

# sum up the total number of times a depthmap value is in the dataset
cm_err_sum = np.sum(cm_data, axis=1)
cm_err = 100 * np.divide(cm_data, cm_err_sum, out=np.zeros(cm_data.shape, dtype=np.float32), where=(cm_err_sum != 0)).reshape(-1)





##-----------------------------------------------------------------------------
def blues(n):

    index = np.transpose(np.arange(n) * (12 / (n-1)))

    color_map = np.empty((n, 3), dtype=np.uint8)

    color_map[:, 0] = np.floor(255*(-0.0724*index + 0.93)).astype(np.uint8)
    color_map[:, 1] = np.floor(255*(-0.0541*index + 0.95)).astype(np.uint8)
    color_map[:, 2] = np.floor(255*(-0.0350*index + 1.00)).astype(np.uint8)

    return color_map


def rgb2hex(data):
    data = np.maximum(0, np.minimum(data, 255))

    cm = []

    for idx in range(data.shape[0]):
        # print("#{0:02X}{1:02x}{2:02x}".format(data[idx, 0], data[idx, 1], data[idx, 2]))
        cm.append("#{0:02X}{1:02x}{2:02x}".format(data[idx, 0], data[idx, 1], data[idx, 2]))

    return cm

def update():

    bp = 1

##-----------------------------------------------------------------------------

# color palettes for plotting
cm_colors = rgb2hex(blues(100))
cm_mapper = LinearColorMapper(palette=cm_colors, low=cm_min, high=cm_max)


# cm_source = ColumnDataSource(data=dict(Predicted=[], Actual=[],  color_value=[]))
cm_source = ColumnDataSource(data=dict(predicted=cm_x, actual=np.flip(cm_y),  value=np.hstack(cm_data), color_value=np.hstack(cm_err)))

file_select_btn = Button(label='Select File', width=100)
filename_div = Div(width=800, text="File name: ", style={'font-size': '125%', 'font-weight': 'bold'})

cm_fig = figure(plot_width=cm_plot_w, plot_height=cm_plot_h,
                x_range=dm_values_str, y_range=list(reversed(dm_values_str)),
                tools="save", toolbar_location="right"
                )


cm_fig.rect(x="predicted", y="actual", width=1.0, height=1.0, source=cm_source, fill_alpha=1.0, line_color='black',
            fill_color=transform('color_value', cm_mapper))

cm_fig.text(x="predicted", y="actual", text=str("value"), source=cm_source,
            text_align="center", text_font_size="13px", text_baseline="middle", text_font_style="bold",
            text_color=transform('color_value', text_mapper))

cm_fig.axis.major_tick_line_color = None
cm_fig.grid.grid_line_color = None

# x-axis formatting
# cm_fig.x_range = Range1d(start=0, end=cm_data_size)
cm_fig.xaxis.major_label_text_font_size = "13pt"
cm_fig.xaxis.major_label_text_font_style= "bold"
cm_fig.xaxis.axis_label_text_font_size = "16pt"
cm_fig.xaxis.axis_label_text_font_style = "bold"
cm_fig.xaxis.axis_label = "Predicted Depthmap Values"

# y-axis formatting
# cm_fig.y_range = Range1d(start=0, end=cm_data_size)
cm_fig.yaxis.major_label_text_font_size = "13pt"
cm_fig.yaxis.major_label_text_font_style= "bold"
cm_fig.yaxis.axis_label_text_font_size = "16pt"
cm_fig.yaxis.axis_label_text_font_style = "bold"
cm_fig.yaxis.axis_label = "Actual Depthmap Values"

input_layout = row(Spacer(width=30), file_select_btn, Spacer(width=10), filename_div)
layout = column(input_layout, cm_fig, Spacer(height=20))

show(layout)


bp = 1

