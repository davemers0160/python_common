import numpy as np
import math

# import bokeh
from bokeh import events

from bokeh.io import curdoc, output_file
from bokeh.models import ColumnDataSource, Spinner, Range1d, Slider, Legend, CustomJS, HoverTool, PointDrawTool, TableColumn, DataTable, NumberFormatter, Div, TextInput
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import column, row, Spacer

from calc_tdoa_position import calc_tdoa_position

"""
<script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-api-2.3.0.min.js" integrity="sha384-RMPdnxafNybXTSOEnNc5DcUZuWp5AI7/X1sevmORhTwgIBG9mS7D1mQ0Fbo2CvCs" crossorigin="anonymous"></script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjs/9.3.2/math.js" integrity="sha512-Imer9iTeuCPbyZUYNTKsBWsAk3m7n1vOgPsAmw4OlkGSS9qK3/WlJZg7wC/9kL7nUUOyb06AYS8DyYQV7ELEbg==" crossorigin="anonymous"></script>
"""
## setup for the GUI
spin_width = 110

# speed of the signal
v = 299792458/1000.0

# predfine the stations
S = 1*np.array([[4, 5],
              [1, 2],
              [-3, 2]], dtype='float')

# time of arrival
T = np.array([0, 0, 0], dtype='float')

# define the location of the target
P = 1*np.array([0, 6], dtype='float')

# define the initital guess
Po = 1*np.array([0.333, 3], dtype='float')

N, num_dim = S.shape

# estimating +/- 10 error
range_err = 0.01

# estimating +/- 0.1us error
time_err = 0.0000001

### ---------------------------------------------------------------------------
p_err_spin = Spinner(title="Position Error (km)", low=0.0, high=10.0, step=0.001, value=range_err, width=spin_width)
t_err_spin = Spinner(title="Time Error (s)", low=0.0, high=1.0, step=0.00000001, value=time_err, width=spin_width)

results_div = Div(width=250)
aou_text = TextInput(title="AOU (km^2)", value="", width=245, background=[255,255,255], disabled=True)


### ---------------------------------------------------------------------------
# calculate the arrival times
def calc_arrival_times(S, T, P, N, v):
    for idx in range(0, N):
        T[idx] = math.sqrt(np.sum((S[idx] - P)*(S[idx] - P)))/v

    return T

### ---------------------------------------------------------------------------
# calculate the covariance matrix and aou circle
def calc_covariance_matrix(P_new, cp, num_trials):
    ## calculate the covariance matrix
    Rp = (1/num_trials) * np.matmul((P_new - cp).transpose(), (P_new - cp))

    # find the eigenvalues (V) and the eigenvectors (E)
    Vp, Ep = np.linalg.eig(Rp)

    # get the confidence interval
    p = 0.95
    s = -2 * math.log(1 - p)
    Vp = Vp * s

    # set the ellipse plotting segments
    theta = np.linspace(0, 2*math.pi, 100)

    # calculate the ellipse
    r_ellipse = np.matmul(np.matmul(Ep, np.sqrt(np.diag(Vp))),  np.vstack((np.cos(theta), np.sin(theta))))
    r_x = r_ellipse[0, :] + cp[0]
    r_y = r_ellipse[1, :] + cp[1]

    aou = np.prod(np.sqrt(Vp))*math.pi

    return r_x, r_y, aou


### ---------------------------------------------------------------------------
# tdoa_source = ColumnDataSource(data=dict(tx=[P[0]], ty=[P[1]], sx=S[:,0], sy=S[:,1], pox=[Po[0]], poy=[Po[1]], v=[v]))
st_source = ColumnDataSource(data=dict(x=(S[:, 0]*1), y=(S[:, 1]*1), t=T, s=S*1, rx=[range_err,range_err,range_err]))
# st_source = ColumnDataSource(data=dict(x=S[:, 0].tolist(), y=S[:, 1].tolist(), t=T.tolist(), rx=[range_err,range_err,range_err]))
ig_source = ColumnDataSource(data=dict(x=[Po[0]*1], y=[Po[1]*1]))
tx_source = ColumnDataSource(data=dict(x=[P[0]*1], y=[P[1]*1]))
etx_source = ColumnDataSource(data=dict(x=[P[0]*1], y=[P[1]*1]))
ctx_source = ColumnDataSource(data=dict(x=[P[0]*1], y=[P[1]*1]))
ell_source = ColumnDataSource(data=dict(x=[P[0]*1], y=[P[1]*1]))


### ---------------------------------------------------------------------------
tdoa_dict = dict(st=st_source, ig=ig_source, ctx=ctx_source, tx=tx_source, etx=etx_source, ell=ell_source, N=N, D=num_dim, v=v, p_err_spin=p_err_spin, t_err_spin=t_err_spin, aou_text=aou_text)
update_plot_callback = CustomJS(args=tdoa_dict, code="""
    
    var num_trials = 100;
    var st_data = st.data;
    
    var v = v;
    var S = []; //st['s'];
    var T = [];
    var Po = [parseFloat(ig.data['x'][0]), parseFloat(ig.data['y'][0])];
    var P = [tx.data['x'][0], tx.data['y'][0] ];

    var range_err = p_err_spin.value;
    var time_err = t_err_spin.value
     
    //--------------------------------------------------------------------
    function randn()
    {
        return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
    }
    
    //--------------------------------------------------------------------
    function calc_tdoa_position(S, T, Po, v)
    {
        // error limit
        var d_err = 1e-3;
        var err = 1000;
    
        // iteration limit
        var max_iter = 50;
        var iter = 0;
        
        // copy the initial guess
        var Pn = Po.slice(0);
        
        // get the min time value and swap the values for S and T
        var t_index = T.indexOf(Math.min(...T)) 
        if(t_index != 0)
        {
            // swap index and 0
            [T[0], T[t_index]] = [T[t_index], T[0]];
            [S[0], S[t_index]] = [S[t_index], S[0]];
        }
        
        //--------------------------------------------------------------------
        while((iter < max_iter) && (err > d_err))
        {
            // calculate the R's
            var R = [];
            for(var idx=0; idx<N; ++idx)
            {
                R[idx] = Math.sqrt( (S[idx][0] - Pn[0])*(S[idx][0] - Pn[0]) + (S[idx][1] - Pn[1])*(S[idx][1] - Pn[1]) );
            }
            
            // build A and b
            var A = [[0,0],[0,0]];
            var b = [0,0];
            for(var idx=1; idx<N; ++idx)
            {
                A[idx - 1][0] = (S[idx][0] - Pn[0])/R[idx] - (S[0][0] - Pn[0])/R[0];
                A[idx - 1][1] = (S[idx][1] - Pn[1])/R[idx] - (S[0][1] - Pn[1])/R[0];
                b[idx - 1] = v * (T[idx] - T[0]) - (R[idx] - R[0]);
            }
                
            var dP = [];          
            var ATA = math.multiply(math.transpose(A), A);
            
            // invertability check
            if(math.det(ATA) == 0)
            {
                //dP = [5,5];
                ATA = math.add(ATA, math.multiply(1.0, math.identity(ATA.length)))._data;
            }

            dP = math.multiply(math.multiply(math.inv(ATA), math.transpose(A)), b)
            
            // generate new Po: Po = Po - dP
            Pn = math.subtract(Pn, dP);
    
            // get the error
            //err = Math.sqrt(dP[0]*dP[0] + dP[1]*dP[1]);
            err = math.sqrt(math.multiply(math.transpose(dP), dP));
                   
            ++iter;
        }
        
        return Pn;
    
    }
       
    //--------------------------------------------------------------------
    function calc_covariance()
    {
        //var CV = mat_mul(P_new[idx])
    }

    //--------------------------------------------------------------------
    // Main function code
    //--------------------------------------------------------------------
    // build S and update the latest arrival time estimate based on the new station positions
    for(var idx = 0; idx<N; idx++)
    {
        S[idx] = [];
        S[idx][0] = st_data['x'][idx];
        S[idx][1] = st_data['y'][idx];
        T[idx] = Math.sqrt( (S[idx][0] - P[0])*(S[idx][0] - P[0]) + (S[idx][1] - P[1])*(S[idx][1] - P[1]) )/v;
    }
      
    var P_new = [];
    var Sn = [];
    var Tn = [];
    
    for(var idx=0; idx<num_trials; ++idx)
    {
    
        for(var jdx=0; jdx<N; ++jdx)
        {
            Sn[jdx] = [S[jdx][0] + range_err*randn(), S[jdx][1] + range_err*randn()];
            Tn[jdx] = T[jdx] + time_err*randn()
        }    
    
        // calculate the position
        P_new[idx] = calc_tdoa_position(Sn, Tn, Po, v);
        
    }
    
    //var cx = arr_avg(P_new.map(function(value,index) { return value[0]; }) );
    //var cy = arr_avg(P_new.map(function(value,index) { return value[1]; }) );
    var C = math.mean(P_new, 0);
    
    var Pm = [];
    Pm[0] = math.subtract(math.row(math.transpose(P_new),0), C[0])[0];
    Pm[1] = math.subtract(math.row(math.transpose(P_new),1), C[1])[0];
    
    // calculate the covariance matrix  
    var Pc = math.multiply(Pm, math.transpose(Pm), 1/num_trials);
    
    // get the eigen values and vectors
    var p_eigs = math.eigs(Pc);
    
    // get the confidence interval
    var scale = -2 * Math.log(1 - 0.95)
    p_eigs.values = math.multiply(p_eigs.values, scale);
    
    // set the ellipse plotting segments
    var theta = math.range(0, 2*Math.PI, 2*Math.PI/100, true);

    // calculate the ellipse
    var r_ellipse = math.multiply(math.multiply(p_eigs.vectors, math.sqrt(math.diag(p_eigs.values))), math.matrix([math.cos(theta), math.sin(theta)]));
    var r_x = math.add(math.row(r_ellipse,0), C[0]);
    var r_y = math.add(math.row(r_ellipse,1), C[1]);
    var aou = math.prod(math.sqrt(p_eigs.values))*Math.PI;
    
    aou_text.value = aou.toPrecision(6);

    // return the results   
    console.log(C);
    
    //st['rx'] = [range_err,range_err,range_err];
    st.data['rx'] = [range_err,range_err,range_err];
    ctx.data['x'] = [C[0]];
    ctx.data['y'] = [C[1]];
    
    etx.data['x'] = P_new.map(function(value,index) { return value[0]; });
    etx.data['y'] = P_new.map(function(value,index) { return value[1]; });
    
    ell.data['x'] = r_x._data[0];
    ell.data['y'] = r_y._data[0];
    
    ctx.change.emit();
    etx.change.emit();
    ell.change.emit();
    st.change.emit();
    
    var bp = 1;
""")


# def update_plot(attr, old, new):
#     st_source.data['rx'] = [p_err_spin.value, p_err_spin.vale, p_err_spin.value]
#
#     bp = 1

# tdoa_source = ColumnDataSource(data=dict(tx=[P[0]], ty=[P[1]], sx=[10], sy=[10], pox=[Po[0]], poy=[Po[1]], v=[v]))
# setup the figure
tdoa_plot = figure(plot_height=700, plot_width=1300, title="TDOA Results")
tdoa_plot.xaxis.axis_label = "X (km)"
tdoa_plot.yaxis.axis_label = "Y (km)"
tdoa_plot.axis.axis_label_text_font_style = "bold"

# tdoa_plot.inverted_triangle(x=(s[0])[:,0], y=(s[0])[:,1], size=5, color='black', source=tdoa_source)
s1 = tdoa_plot.circle(x='x', y='y', radius='rx', fill_color='black', line_color='black', fill_alpha=0.4, source=st_source)
s2 = tdoa_plot.inverted_triangle(x='x', y='y', size=4, color='black', source=st_source)

tdoa_plot.diamond(x='x', y='y', size=10, color='blue', source=ig_source)
tdoa_plot.scatter(x='x', y='y', size=3, color='blue', source=etx_source)
tdoa_plot.scatter(x='x', y='y', size=5, color='lime', source=ctx_source)
tdoa_plot.line(x='x', y='y', line_width=2, color='lime', source=ell_source)
tdoa_plot.scatter(x='x', y='y', size=5, color='red', source=tx_source)
# tool = PointDrawTool(renderers=[s1, s2], num_objects=3)
# tdoa_plot.add_tools(tool)

### ---------------------------------------------------------------------------
# setup the inputs
# define the receiver columns to display
columns = [
    TableColumn(field="x", title="X (km)", formatter=NumberFormatter(format='0[.]000', text_align='center')),
    TableColumn(field="y", title="Y (km)", formatter=NumberFormatter(format='0[.]000', text_align='center')),
]
st_datatable = DataTable(source=st_source, columns=columns, width=250, height=125, editable=True)

# define the initial guess
ig_datatable = DataTable(source=ig_source, columns=columns, width=250, height=75, editable=True)

# define the emitter location
tx_datatable = DataTable(source=tx_source, columns=columns, width=250, height=75, editable=True)


tdoa_results = "<font size='3'>AOU (km^2):<br>"

### ---------------------------------------------------------------------------
T = calc_arrival_times(S, T, P, N, v)


### ---------------------------------------------------------------------------
# set the number of trials
num_trials = 100

P_new = np.zeros([num_trials, num_dim], dtype='float')
iter = np.zeros([num_trials, 1], dtype='float')
err = np.zeros([num_trials, 1], dtype='float')
Sn = np.zeros([N, num_dim, num_trials], dtype='float')

#P_new(idx,:), iter(idx,:), err(idx,:)]= calc_3d_tdoa_position(Sn(:,:, idx), Po, v)
for idx in range(0, num_trials):
    Sn[:, :, idx] = S + np.random.normal(0, range_err, size=(N, num_dim))
    Tn = T + np.random.normal(0, time_err, size=(N))
    P_new[idx], iter[idx], err[idx] = calc_tdoa_position(Sn[:, :, idx], Tn, Po, v)

# get the center/means in each direction
cp = np.mean(P_new, axis=0)

r_x, r_y, aou = calc_covariance_matrix(P_new, cp, num_trials)

# results_div.text = tdoa_results + str(aou) + "</font>"
aou_text.value = "{:.6f}".format(aou)

# st_source.data = dict(sx=Sn[:, 0, :].reshape(-1), sy=Sn[:, 1, :].reshape(-1))
etx_source.data = dict(x=P_new[:, 0]*1, y=P_new[:, 1]*1)
ctx_source.data = dict(x=[cp[0]*1], y=[cp[1]*1])
ell_source.data = dict(x=r_x*1, y=r_y*1)

# setup the event callbacks for the plot
st_source.js_on_change('patching', update_plot_callback)
ig_source.js_on_change('patching', update_plot_callback)
tx_source.js_on_change('patching', update_plot_callback)
p_err_spin.js_on_change('value', update_plot_callback)
t_err_spin.js_on_change('value', update_plot_callback)

# p_err_spin.on_change('value', update_plot)

spin_inputs = row([p_err_spin, Spacer(width=15), t_err_spin])

inputs = column([Div(text="""<B>Receiver Positions</B>""", width=220), st_datatable, Spacer(height=10), Div(text="""<B>Initial Guess</B>""", width=220),
                 ig_datatable, Spacer(height=10), Div(text="""<B>Emitter Position</B>""", width=220), tx_datatable, spin_inputs,
                 Spacer(height=10), aou_text])
layout = row(inputs, Spacer(width=20), tdoa_plot)

show(layout)
