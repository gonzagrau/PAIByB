import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt


def Wavelet(image_path:str, 
            wavelet:str = 'haar', 
            mode:str= 'periodization', 
            Plot_all_levels:bool = True,
            Plot_reconstructed:bool = True):
    "La función pretende calcular la transformada de Wvelets 2D de una imagen segun nivel y ventabna variables"
    #1 importo imag
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #para que sirva tiene que estar en float
    image = np.float32(image)

    #2 hago wavelet 
    
    coeffs2 = pywt.dwt2(image, wavelet,mode )
    cA, (cH, cV, cD) = coeffs2

    #ploteo 
    if Plot_all_levels:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.imshow(cA, cmap='gray')
        plt.title('Approximation Coefficients')

        plt.subplot(2, 2, 2)
        plt.imshow(cH, cmap='gray')
        plt.title('Horizontal Coefficients')

        plt.subplot(2, 2, 3)
        plt.imshow(cV, cmap='gray')
        plt.title('Vertical Coefficients')

        plt.subplot(2, 2, 4)
        plt.imshow(cD, cmap='gray')
        plt.title('Diagonal Coefficients')

        plt.show()

    reconstructed_image = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
    reconstructed_image = np.uint8(reconstructed_image)

    if Plot_reconstructed:
        cv2.imshow('Reconstructed Image', reconstructed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return reconstructed_image


def Wavelet2(image_path:str, 
             wavelet:str = 'haar', 
             mode:str= 'periodization',
             Level:int = 2, 
             Plot_all_levels:bool = True,
             Plot_reconstructed:bool = True):
    "La función pretende calcular la transformada de Wvelets 2D de una imagen segun nivel y ventabna variables"
    #1 importo imag
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #para que sirva tiene que estar en float
    image = np.float32(image)

    #2 hago wavelet 
    coeffs = pywt.wavedec2(image,wavelet,mode,Level)
    

    #ploteo 
    if Plot_all_levels:
       coeff_arr,coefs_slices = pywt.coeffs_to_array(coeffs,)
       plt.figure(figsize=(20,20))
       plt.imshow(coeff_arr,cmap = plt.cm.gray )
       plt.title('Levels of wavelets')
       plt.show()

    reconstructed_image = pywt.waverec2(coeffs,wavelet,mode)
    reconstructed_image = np.uint8(reconstructed_image)
    if Plot_reconstructed:
        cv2.imshow('Reconstructed Image', reconstructed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return reconstructed_image,coeffs

import cv2
import numpy as np
import pywt

def read_coef(Coeffs: tuple, 
              level: int, 
              plot: bool = True, 
              wavelet: str = 'haar', 
              mode: str = 'periodization',
              coef_a_mostrar_vhd:str = 'v'):

    #vector vacio
    modified_coeffs = [np.zeros_like(c) for c in Coeffs]

    if level <= len(Coeffs) - 1:
        cH, cV, cD = Coeffs[-level]
        if coef_a_mostrar_vhd == 'v':
            modified_coeffs[-level] = (np.zeros_like(cH), cV, np.zeros_like(cD))
        elif coef_a_mostrar_vhd == 'h':           
            modified_coeffs[-level] = (cH, np.zeros_like(np.zeros_like(cV)), np.zeros_like(cD))
        else:           
            modified_coeffs[-level] = (np.zeros_like(cH), np.zeros_like(np.zeros_like(cV)),cD)

    else:
        raise ValueError(f"nivel invalido: {level} o coeficiente {coef_a_mostrar_vhd}.")

    recon = pywt.waverec2(modified_coeffs, wavelet, mode)
    recon = np.uint8(np.clip(recon, 0, 255))

    if plot:
        cv2.imshow('Reconstructed Image from Vertical Coefficients', recon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def main():
    a, b =Wavelet2(image_path =r'PAIByB-2\Pie2-1.tif',Level= 1,Plot_all_levels=False,Plot_reconstructed=False)
    #b =Wavelet(image_path =r'GUIA1\PAIByB-2\Pie2-1.tif')
    read_coef(b,level=1,plot=True,coef_a_mostrar_vhd='v')
    
if __name__ == '__main__':
    main()