import numpy as np
import struct
import cv2


from data_handler import create_spectrogram, send_data, receive_data

# Example: Generate some sample IQ data (replace with your actual data)
sample_rate = 10e6  # Sample rate in Hz
duration = 0.01  # Duration in seconds
t = np.arange(0, duration, 1 / sample_rate)
iq_data = np.exp(1j * 2 * np.pi * 1e6 * t)  # Example: a tone at 100 kHz
fft_size = 1024

spectrogram_img = create_spectrogram(iq_data, sample_rate, fft_size)

# Display the image in a window
cv2.imshow("Original Image", spectrogram_img)
# Wait indefinitely until a key is pressed to close the window
cv2.waitKey(0)

# example of slicing an image and "sending" i.e.saving the image
save_file = 'test_spectrogram_img.dat'
send_data(spectrogram_img, save_file)

bp = 1

img_rows = 194
img_width = 3206 # take from the reed solomon result of encoding a line
spectrogram_img2 = receive_data(fft_size, img_width, img_rows, save_file)

# Display the image in a window
cv2.imshow("Decoded Image", spectrogram_img2)
# Wait indefinitely until a key is pressed to close the window
cv2.waitKey(0)

bp = 2

# Destroy all OpenCV windows
cv2.destroyAllWindows()
