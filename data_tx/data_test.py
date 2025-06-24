import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from reedsolo import RSCodec
from matplotlib import cm
import struct

# Example: Generate some sample IQ data (replace with your actual data)
sample_rate = 10e6  # Sample rate in Hz
duration = 0.01  # Duration in seconds
t = np.arange(0, duration, 1 / sample_rate)
iq_data = np.exp(1j * 2 * np.pi * 1e6 * t)  # Example: a tone at 100 kHz

frequencies, times, Sxx = signal.spectrogram(iq_data, fs=sample_rate,
                                            window='hann',
                                            nperseg=1024,
                                            noverlap=512,
                                            return_onesided=False,
                                            mode='magnitude'
                                            )

# Convert to dB (optional, but common for spectrogram visualization)
Sxx_db = 10 * np.log10(Sxx.T + 1e-10)  # Add a small epsilon to avoid log(0)

Sxx_shifted = np.fft.fftshift(Sxx, axes=0).T

# Normalize the data to 0-1 range (for image display)
Sxx_normalized = ((Sxx_shifted - Sxx_shifted.min()) / (Sxx_shifted.max() - Sxx_shifted.min()))

# Scale to 0-255 for 8-bit image representation
spectrogram_image_array = (Sxx_normalized * 255).astype(np.uint8)

#    The colormap returns RGBA values in the 0-1 range.
color_mapped_array = cm.viridis(spectrogram_image_array)

#    Remove the alpha channel if not needed for RGB image.
rgb_image_array = (color_mapped_array[:, :, :3] * 255).astype(np.uint8)

total_rows = rgb_image_array.shape[0]

# Initialize RSCodec with correction length (number of parity bytes)
# This example adds 10 parity bytes, up to 5 lost bytes can be recovered
rs = RSCodec(10)

# loop through each line of the image and pack into a 1d array with row_num/size

for row_index, image_row in enumerate(rgb_image_array):
    print(f"\nProcessing Row {row_index}:")
    # Each 'row' itself is a NumPy array representing a single row of pixels
    # with dimensions (width, channels)
    flattened_image_reshape = image_row.reshape(-1)

    # convert row_index and total rows to byte array
    img_info = np.array([row_index, total_rows]).astype(np.uint16)
    img_info = struct.pack("<2H", *img_info)

    img_info += flattened_image_reshape.tobytes()

    # Encode the data
    encoded_data = rs.encode(img_info)

    img_h = 500
    img_width = 1024

    # Decode the data
    decoded_data, _, errors = rs.decode(encoded_data)

    img_info2 = struct.unpack("<2H", decoded_data[0:4])


