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

def read_coef(Coeffs:tuple,
              level:int,plot:bool = True,
              wavelet:str = 'haar', 
              mode:str= 'periodization'):
    
    tupple_c = Coeffs[-level]
    recon = pywt.waverec2(tupple_c,wavelet,mode)
    recon = np.uint8(recon)
    cv2.imshow('rec', recon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    
    return tupple_c



def main():
    a, b =Wavelet2(image_path =r'C:\Users\Usuario\Desktop\PIByB\PAIByB\GUIA1\PAIByB-2\Pie2-1.tif',Level= 1)
    #b =Wavelet(image_path =r'GUIA1\PAIByB-2\Pie2-1.tif')
    image = cv2.imread(r'C:\Users\Usuario\Desktop\PIByB\PAIByB\GUIA1\PAIByB-2\Pie2-1.tif', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Reconstructed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



if __name__ == '__main__':
    main()