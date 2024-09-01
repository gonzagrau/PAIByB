import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os


def Wavelet(image_path:str, 
            wavelet:str = 'haar', 
            mode:str= 'periodization', 
            Plot_all_levels:bool = True,
            Plot_reconstructed_imagestructed:bool = True):
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

    reconstructed_imagestructed_image = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
    reconstructed_imagestructed_image = np.uint8(reconstructed_imagestructed_image)

    if Plot_reconstructed_imagestructed:
        cv2.imshow('reconstructed_imagestructed Image', reconstructed_imagestructed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return reconstructed_imagestructed_image


def Wavelet2(image_path:str, 
             wavelet:str = 'haar', 
             mode:str= 'periodization',
             Level:int = 2, 
             Plot_all_levels:bool = True,
             Plot_reconstructed_imagestructed:bool = True):
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
       plt.title(f'Level {Level} de wavelets para el archivo {image_path}')
       plt.show()

    reconstructed_imagestructed_image = pywt.waverec2(coeffs,wavelet,mode)
    reconstructed_imagestructed_image = np.uint8(reconstructed_imagestructed_image)
    if Plot_reconstructed_imagestructed:
        cv2.imshow(f'Imagen reconstructed_imagestruida para Level {Level} de wavelets para el arcghivo {image_path}', reconstructed_imagestructed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return reconstructed_imagestructed_image,coeffs

def combine_noise_components(level:int,coeffs, wavelet='haar', mode='periodization'):
    noise_coeffs = [np.zeros_like(coeffs[0])]  # Zero al coef principal

    for i in range(1, len(coeffs)):
        cH, cV, cD = coeffs[i]
        noise_coeffs.append((cH, cV, cD))  # incorporo coefs ruido

    
    return noise_coeffs   

def read_coef(image_path:str,
              Coeffs: tuple, 
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
            modified_coeffs[-level] = (cH, np.zeros_like(cV), np.zeros_like(cD))
        elif coef_a_mostrar_vhd == 'd':            
            modified_coeffs[-level] = (np.zeros_like(cH), np.zeros_like(cV), cD)
        elif coef_a_mostrar_vhd == 'todos':
            modified_coeffs = combine_noise_components(level=level, coeffs=Coeffs, wavelet=wavelet, mode=mode)
        else:
            raise ValueError(f"Invalid coefficient type: {coef_a_mostrar_vhd}. Use 'v', 'h', 'd', or 'todos'.")

    else:
        raise ValueError(f"nivel invalido: {level} o coeficiente {coef_a_mostrar_vhd}.")

    reconstructed_image = pywt.waverec2(modified_coeffs, wavelet, mode)
    

    if plot:
        """ reconstructed_image = np.uint8(np.clip(reconstructed_image, 0, 255))
        cv2.imshow(f'imagen de {coef_a_mostrar_vhd} componentes', reconstructed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows() """
        reconstructed_image = np.float32(reconstructed_image)
        plt.figure(figsize=(10, 10))
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title(f'Level {level}, comoponente {coef_a_mostrar_vhd}, file {image_path}')
        plt.axis('off')
        plt.show()
    return reconstructed_image,modified_coeffs

def get_file_paths(directorio):
    file_paths = []
    for file in os.listdir(directorio):
        relative_path = os.path.join(directorio, file)
        if os.path.isfile(relative_path):
            file_paths.append(relative_path)
    return file_paths


def main():
    #plot original image
    
    """ image = cv2.imread('PAIByB-2\Pie2-2.tif', cv2.IMREAD_GRAYSCALE) 
    image = np.float32(image) 
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.title(f'imagen a mostrar')
    plt.axis('off')
    plt.show() """

    
    Level = 1
    image_path = 'PAIByB-2\Pie2-2.tif'

    image = cv2.imread('PAIByB-2\Pie2-2.tif', cv2.IMREAD_GRAYSCALE) 
    image = np.float32(image) 
    #plot all coef 
    imag_rec, coeffs = Wavelet2('PAIByB-2\Pie2-2.tif',Level= 1,Plot_all_levels= False, Plot_reconstructed_imagestructed= False)
    coeff_arr,coefs_slices = pywt.coeffs_to_array(coeffs,)
    plt.figure(figsize=(20,20))
    plt.imshow(coeff_arr,cmap = plt.cm.gray )
    plt.title(f'Level {Level} de wavelets para el archivo {image_path}')
    plt.show()
    #plot todo el ruido
    imag_Ruido_todo, coef_ruido =read_coef(image_path= image_path,Coeffs= coeffs,level = Level,plot= False,coef_a_mostrar_vhd='todos')
    plt.figure(figsize=(20,20))
    plt.imshow(imag_Ruido_todo,cmap = plt.cm.gray )
    plt.title(f'Ruido para el archivo {image_path}')
    plt.show()
    #plot rec Vs orig

    plt.figure(figsize=(20,20))
    plt.subplot(1,2,1)
    plt.imshow(imag_rec,cmap = plt.cm.gray )
    plt.title(f'imagen recostruida Level {Level}, archivo {image_path}')

    plt.subplot(1,2,2)
    plt.imshow(image,cmap = plt.cm.gray )
    plt.title(f'imagen original, archivo {image_path}')

    

    plt.show()



    #a, b =Wavelet2(image_path =r'PAIByB-2\Pie2-1.tif',Level= 1,Plot_all_levels=False,Plot_reconstructed_imagestructed=False)
    #b =Wavelet(image_path =r'GUIA1\PAIByB-2\Pie2-1.tif')
    #rec = read_coef(b,level=1,plot=True,coef_a_mostrar_vhd='v')
    
    
if __name__ == '__main__':
    main()