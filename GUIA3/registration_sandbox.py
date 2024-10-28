import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_float
from scipy import fftpack
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from reg_features import Imagen, Registracion, lista_de_paths, agrupar_paths
from reg_toolkit import peak_SNR, getPathfiles


def main():
    base_dir = 'PAIByB-5'
    #extrigo paths
    lista_paths = list(getPathfiles(base_dir).values())
    print(lista_paths)
    # Agrupar los paths
    un_digito, dos_digitos = agrupar_paths(lista_paths)
    #aplico la clase Imagen, esta me lee la imagen y calcula los puntos calve
    un_digito_imgs = [Imagen(image, feature_extractor='harris') for image in un_digito ]
    dos_digito_imgs = [Imagen(image, feature_extractor='harris') for image in dos_digitos ]
    print(un_digito)
    print(un_digito)
    for img in dos_digito_imgs[1:]:
        print('\n' + 80 * '_')
        print(f'Imagen {img.nombre}')
        # Crea una instancia de Registracion utilizando Harris como descriptor
        prueba = Registracion(imagen_referencia=dos_digito_imgs[0],
                              imagen_movil=img,
                              lowe_threshold=1,
                              ransac_thres=0,
                              modo='intensity',
                              min_match_count=4)
        prueba.run_pipeline()


if __name__ == '__main__':
    main()