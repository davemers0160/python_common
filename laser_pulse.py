import numpy as np
import math
#import cv2 as cv
from bokeh import events
# import bokeh
from bokeh.io import curdoc, output_file
from bokeh.models import ColumnDataSource, Spinner, Range1d, Slider, Legend, CustomJS, HoverTool
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row, Spacer


## setup some variables
sample_time = 1         # ps
pulse_width = 10        # ps
pri = 10000             # ps
offset = 1000           # ps
num_pulses = 2
wavelength = 350        # nm
f =

p1 = np.full((num_pulses*pri), 0.0, dtype=np.float)
p2 = np.full((num_pulses*pri), 0.0, dtype=np.float)
x = np.arange(0, num_pulses*pri, sample_time)


for idx in range(num_pulses):
    off = offset + (idx*pri)
    p1[off:off+pulse_width] = 1.0
    p2[off:off+pulse_width] = 1.0

src_1 = ColumnDataSource(data=dict(x=[x, x], p=[p1, p2], color=['blue', 'green'], legend=["P1", "P2"]))

p_off = Slider(title="Pulse Offset (ps):", start=0, end=5000, step=1, value=2000, width=1000, callback_policy="mouseup", callback_throttle=50)

pulse_plot = figure(plot_height=300, plot_width=1000, title="Pulse Plot")
pulse_plot.multi_line(xs='x', ys='p', source=src_1, line_width=2, color='color', legend='legend')
pulse_plot.xaxis.axis_label = "Time (ps)"
pulse_plot.yaxis.axis_label = "Pulse Amplitude"
pulse_plot.axis.axis_label_text_font_style = "bold"
pulse_plot.x_range = Range1d(start=0, end=num_pulses*pri)


# Custom JS code to update the plots
cb_dict = dict(source=src_1, pulse_plot=pulse_plot, p_off=p_off, pulse_width=pulse_width, pri=pri, offset=offset, num_pulses=num_pulses)
update_plot_callback = CustomJS(args=cb_dict, code="""
    var data = source.data;
      
    data['p']= [];

    var off = p_off.value;
    
    var p1 = [];
    var p2 = [];

    console.log(off);

    for(var idx = 0; idx<num_pulses; idx++)
    {
        //off = off + offset;
        for(var jdx = 0; jdx<pri; jdx++)
        {
            if(jdx < offset || jdx > offset + pulse_width)
            {
                p1.push(0);
            }
            else
            {
                p1.push(1);
            }
                       
            if(jdx < off || jdx > off + pulse_width)
            {
                p2.push(0);
            }
            else
            {
                p2.push(1);
            }           
        }
    }

    data['p'].push(p1);
    data['p'].push(p2);
    source.change.emit();
""")

def update_plot(attr, old, new):
    global p1, p2
    p2 = np.full((num_pulses * pri), 0.0, dtype=np.float)

    off = p_off.value
    for idx in range(num_pulses):
        off = off + (idx * pri)
        p2[off:off + pulse_width] = 1.0

    src_1.data = dict(x=[x, x], p=[p1, p2], color=['blue', 'green'], legend=["P1", "P2"])


for w in [p_off]:
    # w.on_change('value', update_plot)
    w.js_on_change('value', update_plot_callback)

update_plot(0, 0, 0)

inputs = column(p_off)
layout = column(pulse_plot, p_off)

show(layout)

bp = 1

