import numpy as np
import matplotlib.pyplot as plt
from skimage import io  # Correct import for imread

def ideal_high_pass(shape, cutoff):
    rows, cols = shape
    u, v = np.fft.fftfreq(rows)[:, None], np.fft.fftfreq(cols)[None, :]
    D = np.sqrt(u**2 + v**2)
    
    # Create the filter: 0 for low frequencies, 1 for high frequencies
    H = np.ones(shape)
    H[D < cutoff] = 0
    return H

def gaussian_high_pass(shape, cutoff):
    rows, cols = shape
    u, v = np.fft.fftfreq(rows)[:, None], np.fft.fftfreq(cols)[None, :]
    D = np.sqrt(u**2 + v**2)
    
    # Gaussian high-pass filter
    H = 1 - np.exp(-D**2 / (2 * cutoff**2))
    return H

def apply_filter(image, H):
    F = np.fft.fft2(image)          # Compute the 2D Fourier Transform of the image
    F_shifted = np.fft.fftshift(F)  # Shift the zero frequency component to the center
    F_filtered = F_shifted * H      # Apply the high-pass filter in the frequency domain
    F_ishifted = np.fft.ifftshift(F_filtered)  # Shift back the zero frequency component
    filtered_image = np.fft.ifft2(F_ishifted)  # Inverse Fourier Transform to spatial domain
    return np.abs(filtered_image)   # Return the magnitude of the filtered image

# Load a grayscale image
image_path = r'C:\Users\ibajl\Desktop\Pybib\PAIByB\GUIA1\PAIByB-2\Pie2-1.tif'  # Use raw string literal
image = io.imread(image_path, as_gray=True)

# Create the high-pass filter
cutoff = 0.65
H = ideal_high_pass(image.shape, cutoff)

# Apply the filter
filtered_image = apply_filter(image, H)

# Compute the Fourier Transform of the original image and filtered image
F_original = np.fft.fft2(image)
F_filtered = np.fft.fft2(filtered_image)

# Shift zero frequency component to center
F_original_shifted = np.fft.fftshift(F_original)
F_filtered_shifted = np.fft.fftshift(F_filtered)

# Visualize the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(2, 2, 2)
plt.imshow(np.log(1 + np.abs(F_original_shifted)), cmap='gray')
plt.title("Frequency Domain (Original)")

plt.subplot(2, 2, 3)
plt.imshow(np.log(1 + np.abs(F_filtered_shifted)), cmap='gray')
plt.title("Frequency Domain (Filtered)")

plt.subplot(2, 2, 4)
plt.imshow(filtered_image, cmap='gray')
plt.title("Filtered Image")

plt.show()
