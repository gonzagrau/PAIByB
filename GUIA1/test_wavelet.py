import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
def Wavelet(image_path:str, wavelet:str = 'haar', mode:str= 'periodization', Plot_all_levels:bool = True,Plot_reconstructed:bool = True):
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
