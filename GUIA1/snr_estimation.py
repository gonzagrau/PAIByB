from typing import Dict, Tuple
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from skimage.metrics import structural_similarity as ssim


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


def get_snr_metrics(original_img_dir: str,
                    reconstructed_img_dir: str,
                    print_results: bool = True,
                    plot: bool = True) -> pd.DataFrame:
    """
    Compute peak_SNR and SSIM for an original image and its recontructed version
    :param original_img_dir: directory of original image
    :param reconstructed_img_dir: directory of reconstructed images
    :return: dataframe from PSNR and SSIM
    """
    original_files = os.listdir(original_img_dir)
    reconstructed_files = os.listdir(reconstructed_img_dir)
    original_files.sort()

    original_rec_dict = {original_file: reconstructed_file for original_file in original_files
                         for reconstructed_file in reconstructed_files
                         if original_file.split('.')[0] in reconstructed_file}

    metrics_dic = {}
    if plot:
        fig, axs = plt.subplots(len(original_files), 2, figsize=(10, 20))
    i = 0
    for original, reconstructed in original_rec_dict.items():
        img_original = cv2.imread(os.path.join(original_img_dir, original), cv2.IMREAD_GRAYSCALE)
        img_reconstructed = cv2.imread(os.path.join(reconstructed_img_dir, reconstructed), cv2.IMREAD_GRAYSCALE)
        PSNR = peak_SNR(img_original, img_reconstructed)
        SSIM = ssim(img_original, img_reconstructed)

        metrics_dic[original] = (PSNR, SSIM)

        if print_results:
            print(f"Noise estimation for {original}")
            print(f"{PSNR=:.3f}")
            print(f"{SSIM=:.3f}")
            print()

        if plot:
            axs[i, 0].imshow(img_original, cmap='gray', vmin=0, vmax=255)
            axs[i, 0].set_title(f'Original Image {original}')
            axs[i, 1].imshow(img_reconstructed, cmap='gray', vmin=0, vmax=255)
            axs[i, 1].set_title(f'Reconstructed with {PSNR=:.3f}, {SSIM=:.3f}')
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')

        i += 1
    

    if plot:
        plt.axis('off')
        plt.show()

    metrics_df = pd.DataFrame(metrics_dic).T.rename({0: 'PSNR', 1:'SSIM'}, axis='columns')
    return metrics_df
