import numpy as np
from filter_windows import nuttall_window, create_fir_filter


def upsample_data(data, N):
    # upsample data by an integer factor N.  For fractional upsampling upsample
    # by N and downsample by M to reach the correct fractional value

    filter_tap_mult = 20

    num_data_samples = data.shape[0]
    y = np.zeros(num_data_samples*N, np.float32)

    # insert samples with N-1 zeros
    for idx in range(num_data_samples):
        y[idx*N] = data[idx]

    # create fir filter
    n_taps = filter_tap_mult*N
    if (n_taps % 2) == 0:
        n_taps = n_taps + 1

    # filter cutoff frequency
    fc = 1/N

    w = nuttall_window(n_taps)
    lpf = create_fir_filter(fc, w)

    # normalize lpf
    lpf_sum = np.sum(lpf)
    lpf = lpf/lpf_sum

    # convolve with scaled version of the lpf based on the upsampling factor
    data = np.convolve(y, N*lpf[::-1], 'same')

    return data
