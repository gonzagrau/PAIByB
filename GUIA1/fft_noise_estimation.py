import numpy as np
import matplotlib.pyplot as plt

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

a = gaussian_high_pass((10,10),0.5)
print(a)