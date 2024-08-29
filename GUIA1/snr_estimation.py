import numpy as np
import cv2


def peak_SNR(img_og: np.ndarray, img_noise: np.ndarray) -> float:
    """
    Peak signal to noise ratio estimation
    :param img_og: original image
    :param img_noise: noisy image
    :return: peak signal to noise ratio estimation
    """
    assert img_og.shape == img_noise.shape
    L = np.iinfo(img_og.dtype).max
    MSE = np.mean((img_og - img_noise)**2)
    return 10*np.log(L**2/MSE)