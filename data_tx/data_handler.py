import numpy as np
import numpy.typing as npt

from scipy import signal
from reedsolo import RSCodec
from matplotlib import cm
import struct

#------------------------------------------------------------------------------
def create_spectrogram(iq_data, sample_rate, fft_size):

    # create the initial spectrogram in magnitude mode
    frequencies, times, Sxx = signal.spectrogram(iq_data, fs=sample_rate,
                                                 window='hann',
                                                 nperseg=fft_size,
                                                 noverlap=fft_size // 2,
                                                 return_onesided=False,
                                                 mode='magnitude'
                                                 )

    # perform fft shift and transpose
    Sxx_shifted = np.fft.fftshift(Sxx, axes=0).T

    # Normalize the data to 0-1 range (for image display)
    Sxx_normalized = ((Sxx_shifted - Sxx_shifted.min()) / (Sxx_shifted.max() - Sxx_shifted.min()))

    # Scale to 0-255 for 8-bit image representation
    spectrogram_image_array = (Sxx_normalized * 255).astype(np.uint8)

    # The colormap returns RGBA values in the 0-1 range.
    color_mapped_array = cm.jet(spectrogram_image_array)

    # Remove the alpha channel if not needed for RGB image.
    rgb_image_array = (color_mapped_array[:, :, :3] * 255).astype(np.uint8)

    return rgb_image_array

#------------------------------------------------------------------------------
# spectrogram_float (N x M x 1) np array(uint8)
# rows should be time
# cols should be frequency
def send_data(spectrogram_float: npt.NDArray[np.uint8], file=None):

    total_rows = spectrogram_float.shape[0]
    img_width = spectrogram_float.shape[1]

    # Initialize RSCodec with correction length (number of parity bytes)
    # adds 10 parity bytes, up to 5 lost bytes can be recovered
    rs = RSCodec(10)

    # Open the file in binary write mode ('wb')
    if file is not None:
        file_object = open("test_image.bin", "wb")

    for row_index, image_row in enumerate(spectrogram_float):
        print(f"\nProcessing Row {row_index}:")
        # Each 'row' itself is a NumPy array representing a single row of pixels
        # with dimensions (width, channels)
        flattened_image_reshape = image_row.reshape(-1)

        # convert row_index and total rows to byte array
        img_info = np.array([row_index, total_rows]).astype(np.uint16)
        img_info = struct.pack("<2H", *img_info)

        # append the data to the byte array of image info
        img_info += flattened_image_reshape.tobytes()

        # Encode the data
        encoded_img_row = rs.encode(img_info)

        if file is not None:
            # save the data to a file
            file_object.write(encoded_img_row)

        else:
            # send the data using hardware
            bp = 1


    if file is not None:
        file_object.close()

#------------------------------------------------------------------------------
def receive_data(img_width, file=None):
    # Initialize RSCodec with correction length (number of parity bytes)
    # adds 10 parity bytes, up to 5 lost bytes can be recovered
    rs = RSCodec(10)

    # Open the file in binary read mode ('wb')
    if file is not None:
        file_object = open("test_image.bin", "rb")

    # Decode the data
    decoded_data, _, errors = rs.decode(encoded_data)

    img_info2 = struct.unpack("<2H", decoded_data[0:4])
