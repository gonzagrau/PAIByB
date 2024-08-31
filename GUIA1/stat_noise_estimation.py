import os
import numpy as np
import json
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, Dict, Any


def load_stat_data(data_path: str) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Search for saved data from image segment analysis
    :param data_path: directory
    :return: stat_data, with statistical analysis,
             hist_dict, with a dataframe for each image's histogram
    """
    data_dict = {}
    hist_dict = {}
    for image in os.listdir(data_path):
        for file in os.listdir(os.path.join(data_path, image)):
            if file.endswith('.json'):
                data_dict[image] = json.load(open(os.path.join(data_path, image, file), 'r'))
            elif file.endswith('.csv'):
                hist_dict[image] = pd.read_csv(os.path.join(data_path, image, file))

    return data_dict, hist_dict


def stat_noise_reconstruction(data_dict: Dict[str, Any],
                              input_dir: str,
                              output_dir: str) -> None:
    """
    Reconstruct noise from statistical analysis of an image ROI selection
    :param data_dict: returned from load_stat_data
    :param input_dir: image origin dir
    :param output_dir: image reconstruction dir
    :return: None
    """
    for fname in os.listdir(input_dir):
        noise = fname.split('.')[0]
        ruido_real = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
        stats = data_dict[noise]
        ruido_reconstruido = np.random.normal(loc=stats['mean'],
                                              scale=stats['desv_est'],
                                              size=ruido_real.shape)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(ruido_real, vmin=0, vmax=255, cmap='gray')
        axs[0].set_title(f'Imagen original: {noise}')
        axs[1].imshow(ruido_reconstruido, vmin=0, vmax=255, cmap='gray')
        axs[1].set_title('Ruido reconstruido con media y varianza')
        plt.show()

        cv2.imwrite(os.path.join(output_dir, f"{noise}_stat.png"), ruido_reconstruido)


def hist_noise_reconstruction(hist_dict: Dict[str, Any],
                              input_dir: str,
                              output_dir: str) -> None:
    """
    Reconstruct image noise from the histogram of a ROI selection
    :param hist_dict: returned from load_stat_data
    :param input_dir: image origin dir
    :param output_dir: reconstructed image dir
    :return: None
    """

    for fname in os.listdir(input_dir):
        noise = fname.split('.')[0]
        hist_df = hist_dict[noise]
        I, h = hist_df['gray_lvl'], hist_df['hist_value']
        mu = np.dot(I, h) / h.sum()
        sigma = np.sqrt(np.dot((I - mu) ** 2, h) / h.sum())
        bell_curve = norm.pdf(x=I, loc=mu, scale=sigma) * h.sum()

        fig, ax = plt.subplots()
        ax.plot(hist_df['gray_lvl'], hist_df['hist_value'])
        ax.plot(hist_df['gray_lvl'], bell_curve)
        ax.set_title(f'Histograma vs. ajuste gaussiano para {noise}')
        plt.show()

        # Calculo DMA
        DMA = np.abs(I - mu).dot(h) / h.sum()
        print(f"{DMA=:.3f}")

        # Reconstruimos
        ruido_real = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
        ruido_reconstruido = np.random.normal(loc=mu,
                                              scale=sigma,
                                              size=ruido_real.shape)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(ruido_real, vmin=0, vmax=255, cmap='gray')
        axs[0].set_title(f'Imagen original: {noise}')
        axs[1].imshow(ruido_reconstruido, vmin=0, vmax=255, cmap='gray')
        axs[1].set_title('Ruido reconstruido con el histograma')
        plt.show()

        # Guardamos
        cv2.imwrite(os.path.join(output_dir, f"{noise}_hist.png"), ruido_reconstruido)