import math
import numpy as np

#-----------------------------------------------------------------------------------------------------
def nuttall_window(N):

    w = np.zeros(N)
    a0 = 0.355768
    a1 = 0.487396
    a2 = 0.144232
    a3 = 0.012604

    for idx in range(N):
        w[idx] = a0 - a1 * math.cos(2.0 * np.pi * idx / (N-1)) + a2 * math.cos(4.0 * np.pi * idx / (N-1)) - a3 * math.cos(6.0 * np.pi * idx / (N-1))
    return w

#-----------------------------------------------------------------------------------------------------
def blackman_nuttall_window(N):

    w = np.zeros(N)
    a0 = 0.3635819
    a1 = 0.4891775
    a2 = 0.1365995
    a3 = 0.0106411

    for idx in range(N):
        w[idx] = a0 - a1 * math.cos(2.0 * np.pi * idx / (N-1)) + a2 * math.cos(4.0 * np.pi * idx / (N-1)) - a3 * math.cos(6.0 * np.pi * idx / (N-1))
    return w

#-----------------------------------------------------------------------------------------------------
def create_fir_filter(fc, w):

    N = w.shape[0]
    g = np.zeros(N)

    for idx in range(N):

        x = np.pi * (idx - N / 2.0)

        if (abs(idx - (N / 2.0)) < 1e-6):
            g[idx] = w[idx] * fc
        else:
            g[idx] = w[idx] * (math.sin(x * fc) / x)

    return g

