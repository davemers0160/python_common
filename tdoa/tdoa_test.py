
import math
import numpy as np
from calc_tdoa_position import calc_tdoa_position
# import plotly.express as px
import plotly.graph_objects as go

v = 299792458

# set the station locations (S) and the emitter location (P)
# 3D
# S(1,:) = [100, 150, 89, 0]
# S(2,:) = [345, 212, 334, 0]
# S(3,:) = [212, 343, 121, 0]
# S(4,:) = [260, 564, 33, 0]
# P = [200, 100, 0]

# 2D
# S(1,:) = [100, 150, 0]
# S(2,:) = [325, 222, 0]
# S(3,:) = [212, 303, 0]

# S(1,:) = [145, 250, 0]
# S(2,:) = [200, 250, 0]
# S(3,:) = [250, 250, 0]

S = np.array([[3, 5, 0],
              [1, 2, 0],
              [-3, 2, 0]], dtype='float')


#S(4,:) = [255, 564, 0]
P = np.array([0, 6], dtype='float')

N, num_dim = S.shape
num_dim = num_dim - 1

# calculate the arrival times
for idx in range(0, N):
    S[idx, -1] = math.sqrt(np.sum((S[idx, 0:-1] - P)*(S[idx, 0:-1] - P)))/v


# guess/calculate an initial position
# Po = [250, 150, 0]
Po = np.array([0.333, 3], dtype='float')

# set the number of trials
num_trials = 100

P_new = np.zeros([num_trials, num_dim], dtype='float')
iter = np.zeros([num_trials, 1], dtype='float')
err = np.zeros([num_trials, 1], dtype='float')
Sn = np.zeros([N, num_dim+1, num_trials], dtype='float')

#P_new(idx,:), iter(idx,:), err(idx,:)]= calc_3d_tdoa_position(Sn(:,:, idx), Po, v)
for idx in range(0, num_trials):
    Sn[:, 0:-1, idx] = S[:, 0:-1] + np.random.normal(0, 0.01, size=(N, num_dim))
    Sn[:, -1, idx] = S[:, -1] + np.random.normal(0, 0.0000000001, size=(N))
    P_new[idx], iter[idx], err[idx] = calc_tdoa_position(Sn[:, :, idx], Po, v)

# get the center/means in each direction
cp = np.mean(P_new, axis=0)

bp = 1

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

## plot everything
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=Sn[:, 0, :].reshape(-1), y=Sn[:, 1, :].reshape(-1),
            mode='markers',
            name='Receivers',
            marker_color='rgba(0, 0, 0, 1.0)',
            marker_symbol='triangle-down'))

fig.add_trace(go.Scatter(x=[Po[0]], y=[Po[1]],
                         mode='markers',
                         name='Initial Guess',
                         marker_color='rgba(0, 0, 255, 1.0)',
                         marker_symbol='diamond'))

fig.add_trace(go.Scatter(x=P_new[:, 0], y=P_new[:, 1],
                         mode='markers',
                         name='Transmitter',
                         marker_size=3,
                         marker_color='rgba(0, 0, 255, 1.0)',
                         marker_symbol='circle',
                         ))

fig.add_trace(go.Scatter(x=r_x, y=r_y,
                         mode='lines',
                         name='error',
                         line_width=1,
                         marker_color='rgba(0, 255, 0, 1.0)',
                         ))

fig.add_trace(go.Scatter(x=[cp[0]], y=[cp[1]],
                         mode='markers',
                         name='Transmitter',
                         marker_color='rgba(255, 0, 0, 1.0)',
                         marker_symbol='circle',
                         ))


fig.show()

bp = 1
