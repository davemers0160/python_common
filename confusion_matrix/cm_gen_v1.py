import os
import math
import numpy as np

from bokeh import events
from bokeh.io import curdoc, output_file
from bokeh.models import ColumnDataSource, Button, Div, Legend, CustomJS, HoverTool, LinearColorMapper, CategoricalColorMapper, Range1d
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

# path to start looking in for confusion matrix files
start_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# get the min and max for the color shading of the confusion matrix.  Min is assumed to be 0.
cm_min = 0
cm_max = 100

# plot specific variables that are used in plot formatting (size/color)...
cm_plot_h = 750
cm_plot_w = 2100
err_plot_h = 130
err_plot_w = cm_plot_w

dm_values_str = [str(x) for x in range(0, 2)]
cm_data = np.array([[1, 0], [0, 1]])

cm_source = []
cm_err_source = []
hist_source = []

cm_fig = figure(plot_width=cm_plot_w, plot_height=cm_plot_h,
                x_range=dm_values_str, y_range=list(reversed(dm_values_str)),
                tools="save", toolbar_location="right"
                )

err_fig = figure(plot_width=err_plot_w, plot_height=err_plot_h,
                 y_range="1", x_range=dm_values_str,
                 tools="save", toolbar_location="below"
                 )

# hover = HoverTool(tooltips=[("Value: ", "$x"),  ("Count: ", "$data")], point_policy="snap_to_data", line_policy="nearest", mode="mouse")
hist_fig = figure(plot_width=cm_plot_w, plot_height=cm_plot_h,
                 tooltips=[('Value:', '@x{0}'), ('Count:', '@data{0}')],
                 tools="save, pan, box_zoom, reset, wheel_zoom, hover", toolbar_location="right"
                 )

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


def get_input():
    global start_path, filename_div   #, results_div, filename_div, image_path, rgba_img

    file_name = QFileDialog.getOpenFileName(None, "Select a confusion matrix csv file",  start_path, "Text Files (*.txt);;CSV Files (*.csv);;All Files (*.*)")
    if(file_name[0] == ""):
        return

    filename_text = "File name: " + file_name[0]
    filename_div.text = filename_text

    print("Processing File: ", file_name[0])
    start_path = os.path.dirname(file_name[0])

    # load the data
    cm_data = pd.read_csv(file_name[0], header=None).values

    build_dataframes(cm_data)

    # run_detection(color_img)
    update_plot()

    return cm_data, filename_text


def build_dataframes(cm_data):
    global cm_source, cm_err_source, dm_values_str, hist_source

    # cm_data_size = cm_data.shape[0]

    dm_min = 0
    dm_max = cm_data.shape[0] - 1
    dm_values_str = [str(x) for x in range(dm_min, dm_max+1)]

    # sum up the total number of times a depthmap value is in the dataset
    cm_err_sum = np.sum(cm_data, axis=1)

    # format the data for a histogram view, use cm_err_sum for the data portion
    bins = np.linspace(dm_min, dm_max, dm_max+1)
    hist_source = ColumnDataSource(data=dict(x=bins, data=cm_err_sum))

    # calculate how many times the prediction is correct
    cm_err_diag = np.diag(cm_data)
    cm_err_data = 100 * np.divide(cm_err_diag, cm_err_sum, out=np.zeros(cm_err_diag.shape, dtype=np.float32), where=(cm_err_sum != 0))
    cm_err_data = np.subtract(100, cm_err_data, out=np.zeros(cm_err_data.shape, dtype=np.float32), where=(cm_err_data != 0))
    cm_err_cat = 1*(cm_err_data > 5)+1*(cm_err_data > 10)


    cm_df = pd.DataFrame(data=cm_data, index=dm_values_str, columns=dm_values_str)
    cm_df.index.name = "Actual"
    cm_df.columns.name = "Predicted"
    cm_df = cm_df.stack().rename("value").reset_index()
    cm_df['color_value'] = 100 * np.divide(cm_data, cm_err_sum, out=np.zeros(cm_data.shape, dtype=np.float32), where=(cm_err_sum != 0)).reshape(-1)
    cm_source = ColumnDataSource(cm_df)

    # cm_values_str = ["{:4.2f}%".format(cm_err_data[x]) for x in range(dm_min, dm_max+1)]

    cm_err_df = pd.DataFrame(data=cm_err_data.reshape(1, -1), index=["1"], columns=dm_values_str)
    cm_err_df.index.name = 'Error'
    cm_err_df.columns.name = 'Label'
    cm_err_df = cm_err_df.stack().rename("value").reset_index()
    cm_err_df['str_value'] = ["{:4.2f}%".format(cm_err_data[x]) for x in range(dm_min, dm_max+1)]
    cm_err_df['err_cat'] = [str(cm_err_cat[x]) for x in range(dm_min, dm_max+1)]
    cm_err_source = ColumnDataSource(cm_err_df)


def update_plot():
    global cm_source, cm_err_source, hist_source, dm_values_str, cm_fig, err_fig, hist_fig

    # update the cm_fig X and Y values
    cm_fig.x_range.factors = dm_values_str
    cm_fig.y_range.factors = list(reversed(dm_values_str))

    # update the err_fig X values
    err_fig.x_range.factors = dm_values_str

    cm_fig.rect(x="Predicted", y="Actual", width=1.0, height=1.0, source=cm_source, fill_alpha=1.0, line_color='black',
                fill_color=transform('color_value', cm_mapper))

    text_props = {"source": cm_source, "text_align": "center", "text_font_size": "13px", "text_baseline": "middle", "text_font_style": "bold"}

    # x = dodge("Predicted", 0.0, range=cm_fig.x_range)

    cm_fig.text(x="Predicted", y="Actual", text=str("value"), text_color=transform('color_value', text_mapper), **text_props)

    # error figure
    err_fig.rect(x="Label", y="Error", width=1.0, height=1.0, source=cm_err_source, fill_alpha=1.0, line_color='black',
                 fill_color=transform('err_cat', CategoricalColorMapper(palette=error_colors, factors=["0", "1", "2"])))

    err_fig.text(x="Label", y="Error", text="str_value", source=cm_err_source, text_align="center",
                 text_color=transform('err_cat', CategoricalColorMapper(palette=["#000000", "#000000", "#FFFFFF"], factors=["0", "1", "2"])),
                 text_font_size="13px", text_baseline="middle", text_font_style="bold")

    # histogram plot
    hist_fig.renderers = []
    hist_fig.vbar(x="x", top="data", width=0.9, source=hist_source, fill_color="blue", line_color="white")
    hist_fig.y_range.start = 0
    hist_fig.x_range = Range1d(-0.5, 22+0.5)

##-----------------------------------------------------------------------------

file_select_btn = Button(label='Select File', width=90, height=28)
file_select_btn.on_click(get_input)
filename_div = Div(width=1450, text="File name: ", style={'font-size': '125%', 'font-weight': 'bold'})

# color palettes for plotting
cm_colors = rgb2hex(blues(200))
cm_mapper = LinearColorMapper(palette=cm_colors, low=cm_min, high=cm_max)

error_colors = ["#00FF00", "#FFA700", "#FF0000"]

text_mapper = LinearColorMapper(palette=["#000000", "#FFFFFF"], low=75, high=cm_max)

get_input()
# build_dataframes(cm_data)
# update_plot()

# confusion matrix figure formatting
cm_fig.axis.major_tick_line_color = None
cm_fig.grid.grid_line_color = None

# x-axis formatting
cm_fig.xaxis.major_label_text_font_size = "13pt"
cm_fig.xaxis.major_label_text_font_style= "bold"
cm_fig.xaxis.axis_label_text_font_size = "16pt"
cm_fig.xaxis.axis_label_text_font_style = "bold"
cm_fig.xaxis.axis_label = "Predicted Depthmap Values"

# y-axis formatting
cm_fig.yaxis.major_label_text_font_size = "13pt"
cm_fig.yaxis.major_label_text_font_style= "bold"
cm_fig.yaxis.axis_label_text_font_size = "16pt"
cm_fig.yaxis.axis_label_text_font_style = "bold"
cm_fig.yaxis.axis_label = "Actual Depthmap Values"

# error figure formatting
err_fig.axis.major_tick_line_color = None
err_fig.grid.grid_line_color = None
err_fig.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels
err_fig.xaxis.major_label_text_font_size = "13pt"
err_fig.xaxis.major_label_text_font_style= "bold"
err_fig.xaxis.axis_label_text_font_size = "16pt"
err_fig.xaxis.axis_label_text_font_style = "bold"
err_fig.xaxis.axis_label = "Actual Depthmap Errors"

# histogram figure formatting
hist_fig.xaxis.major_label_text_font_size = "13pt"
hist_fig.xaxis.major_label_text_font_style= "bold"
hist_fig.xaxis.axis_label_text_font_size = "16pt"
hist_fig.xaxis.axis_label_text_font_style = "bold"
hist_fig.xaxis.axis_label = "Depthmap Value"

hist_fig.yaxis.major_label_text_font_size = "13pt"
hist_fig.yaxis.major_label_text_font_style = "bold"
hist_fig.yaxis.axis_label_text_font_size = "16pt"
hist_fig.yaxis.axis_label_text_font_style = "bold"
hist_fig.yaxis.axis_label = "Depthmap Count"

# layout
input_layout = row(Spacer(width=30), file_select_btn, Spacer(width=10), filename_div)
# input_layout = row(Spacer(width=50), filename_div)
layout = column(input_layout, cm_fig, Spacer(height=15), err_fig, Spacer(height=15), hist_fig)

# show(layout)

doc = curdoc()
doc.title = "Confusion Matrix Viewer"
doc.add_root(layout)

bp = 1
