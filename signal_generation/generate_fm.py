import numpy as np
import math
from upsample_data import upsample_data


def generate_fm(data, audio_fs, rf_fs, k):

    # interpolation scale multiplier
    N = math.floor(rf_fs/audio_fs)
    sample_rate = N * audio_fs
    num_data_samples = data.shape[0]
    num_rf_samples = math.floor(num_data_samples * N)

    scale = 2 * np.pi * 1j * k

    iq = np.zeros(num_rf_samples, np.cdouble)

    # upsample data
    y2 = upsample_data(data, N)

    # shift data to approximately zero mean
    y2_mean = np.mean(y2)
    y2 = y2 - y2_mean

    accum = 0

    # apply FM modulation
    y_accum = np.cumsum(y2)
    iq = np.exp(scale * y_accum)

    # for idx in range(num_rf_samples):
    #     accum += y2[idx]
    #     iq[idx] =

    return iq, sample_rate, y2
