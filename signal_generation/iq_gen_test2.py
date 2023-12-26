import numpy as np
import soundfile as sf

from generate_fm import generate_fm
from iq_utils.binary_file_ops import write_binary_iq_data

filename = 'd:/data/humpback_whale.mp3'
filename = 'd:/music/Metallica/Metallica - For Whom The Bell Tolls (LP Version).mp3'

y, fs = sf.read(filename)

# grab only one channel
if (y.ndim > 1):
    y = np.mean(y, axis=1)

[iq, sample_rate, y2] = generate_fm(0.75*y, fs, 20*fs, 0.02)

write_binary_iq_data('d:/data/bells_882K_2.sc16', np.round(1950*iq))
