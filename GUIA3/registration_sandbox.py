import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_float
from scipy import fftpack
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from registration import Imagen, Registracion, lista_de_paths, agrupar_paths
from reg_toolkit import peak_SNR, getPathfiles, matchImg

def test_luki():
    img_template_path = 'PAIByB-5/img-1.tif'
    img_template = cv2.imread(img_template_path, cv2.IMREAD_GRAYSCALE)
    img_other_path = 'PAIByB-6/img-2.tif'
    img_other = cv2.imread(img_other_path, cv2.IMREAD_GRAYSCALE)
    reg, data = matchImg(img_template, img_other)



def main():
    base_dir = 'PAIByB-5'
    #extrigo paths
    lista_paths = list(getPathfiles(base_dir).values())
    # Agrupar los paths
    un_digito, dos_digitos = agrupar_paths(lista_paths)
    #aplico la clase Imagen, esta me lee la imagen y calcula los puntos calve
    un_digito_imgs = [Imagen(image) for image in un_digito ]
    dos_digito_imgs = [Imagen(image) for image in dos_digitos ]

    fixed_img = un_digito_imgs[0]
    for img in un_digito_imgs[1:]:
        print('\n' + 80 * '_')
        print(f'Imagen {img.nombre} vs. {fixed_img.nombre}')

        try:
            matchImg(img.imagen, fixed_img.imagen)
            print('match correcto con la de luki')

        except Exception as e:
            print(e)
            continue

        # Crea una instancia de Registracion utilizando Harris como descriptor
        prueba = Registracion(imagen_referencia=fixed_img, imagen_movil=img)
        prueba.intensity_registration(mode=1)
        prueba.plot_registration()


if __name__ == '__main__':
    main()