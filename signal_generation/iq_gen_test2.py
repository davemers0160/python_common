"""
requirements: the primary requirement for this code to run is the python library soundfile.  

Windows: Easy, just pip install

pip install soundfile >= 0.12.1

Linux: not so easy.  Depending on your OS and or processor (Intel, ARM, etc...) the right libsndfile1 library may not be available.  In order to read mp3 file the libsndfile1 library must be >= v1.2.0
To check the availabe version run this command: $ apt list | grep libsndfile1
You can try to force the install: $ sudo apt-get install libsndfile1 >= 1.2.0

"""

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

# this assumes that the data comes in the [-1, 1) range
y, fs = sf.read(filename)

# if the audio file is a 2-channel stereo steam then we just average the stream 
if (y.ndim > 1):
    y = np.mean(y, axis=1)

# take the audio and upsample 
[iq, sample_rate, y2] = generate_fm(y, fs, 20*fs, 0.1)

# write the IQ data to a binary file
write_binary_iq_data('d:/data/whale_882K.sc16', np.round(1950*iq))
