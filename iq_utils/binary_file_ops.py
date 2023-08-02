import numpy as np
import struct
import os

''' Write numpy array to uint32 binary data file '''
def write_binary_image(file_name, data):
    file_id = open(file_name, 'wb')
    height = data.shape[0]
    width = data.shape[1]

    file_id.write(struct.pack('<I', height))
    file_id.write(struct.pack('<I', width))

    #data.astype('<u4').tofile(file_id)
    #file_id.write(struct.pack('<I', data))
    np.array(data, dtype=np.uint32).tofile(file_id)

    file_id.close()


''' Read in binary uint32 data from a file in little endian order '''
def read_binary_image(file_name):
    file_id = open(file_name, 'rb')

    height = np.fromfile(file_id, '<u4', 1, "")
    width = np.fromfile(file_id, '<u4', 1, "")

    data = np.fromfile(file_id, '<u4', -1, "")

    data = data.reshape(height[0], width[0])

    file_id.close()

    return data
    
    