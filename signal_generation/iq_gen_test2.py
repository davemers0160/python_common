import os
import sys
import numpy as np
import soundfile as sf

from generate_fm import generate_fm

## do this because the relative path doesn't work on linux without PYTHONPATH
script_path = os.path.realpath(__file__)
project_path = os.path.dirname(os.path.dirname(script_path))
sys.path.insert(0, os.path.abspath(project_path))
from iq_utils.binary_file_ops import write_binary_iq_data

filename = 'd:/data/humpback_whale.mp3'
# filename = 'd:/music/Metallica/Metallica - For Whom The Bell Tolls (LP Version).mp3'

y, fs = sf.read(filename)

# grab only one channel
if (y.ndim > 1):
    y = np.mean(y, axis=1)

[iq, sample_rate, y2] = generate_fm(y, fs, 20*fs, 0.1)

write_binary_iq_data('d:/data/whale_882K.sc16', np.round(1950*iq))
