# II. Analice las texturas del contorno elegido mediante filtros Gabor, transformada de Fuorier y
# transformada wavelet. ¿Puede caracterizar de forma única la textura de los objeto mediante
# los métodos utilizados?¿Qué puede decir de los valores obtenidos para cada imagen? """
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import os

def caracteristicas_fourier(imagen,plot:bool = False):
    #aplico la transformada de fourier y centro 
    f_trans = np.fft.fft2(imagen)
    f_shift = np.fft.fftshift(f_trans)
    mag_esp = 20 * np.log(np.abs(f_shift) + 1)

    energia = np.sum(np.abs(f_shift))

    #Divido la transformada en cuadrantes
    h, w = f_shift.shape
    c_h, c_w = h // 2, w // 2
    PerC = f_shift[:c_h, :c_w]
    SdoC = f_shift[:c_h, c_w:]
    TerC = f_shift[c_h:, :c_w]
    CtoC = f_shift[c_h:, c_w:]

    

    ener_1erC = np.sum(np.abs(PerC))
    ener_2doC = np.sum(np.abs(SdoC))
    ener_3erC = np.sum(np.abs(TerC))
    ener_4toC = np.sum(np.abs(CtoC))

    # Almacenar los resultados en un diccionario
    resultados = {
        'Energía Total': energia,
        'Potencia Total': mag_esp,
        'Energía Primer Cuadrante': ener_1erC,       
        'Energía Segundo Cuadrante': ener_2doC,       
        'Energía Tercer Cuadrante': ener_3erC,
        'Energía Cuarto Cuadrante': ener_4toC,
        
    }
    if plot:
        # Mostrar la imagen original y su espectro de Fourier
        plt.figure(figsize=(12, 6))

        # Imagen original
        plt.subplot(1, 2, 1)
        plt.imshow(imagen, cmap='gray')
        plt.title('Imagen Original')

        # Espectro de magnitud de Fourier
        plt.subplot(1, 2, 2)
        plt.imshow(mag_esp, cmap='gray')
        plt.title('Espectro de Magnitud de Fourier')
        plt.show()
    return resultados

def caracteristicas_wavelets(imagen,wavelet:str = 'haar',plot:bool = False, mode: str = 'periodization',Level:int = 2,Plot_all_levels:bool = False ):
    coef = pywt.wavedec2(imagen, wavelet=wavelet, level=Level,mode = mode) 
    cA = coef[0]  # Coeficientes de aproximación (bajas frecuencias)
    cHVD = coef[1:]  # Coeficientes de detalles (altas frecuencias)

    # Energía de la aproximación
    energia_approx = np.sum(np.square(cA))

    # Diccionario para almacenar las energías
    energy_dict = {
        'Energía Aproximación': energia_approx
    }

    # Calcular las energías de los detalles (horizontal, vertical, diagonal) por cada nivel
    for i, (cH, cV, cD) in enumerate(cHVD):
        energia_H = np.sum(np.square(cH))  # Energía del detalle horizontal
        energia_V = np.sum(np.square(cV))  # Energía del detalle vertical
        energia_D = np.sum(np.square(cD))  # Energía del detalle diagonal

        # Agregar la energía por nivel al diccionario
        energy_dict[f'Nivel {i+1}'] = {
            'Energía Horizontal': energia_H,
            'Energía Vertical': energia_V,
            'Energía Diagonal': energia_D
        }
        #ploteo 
    if Plot_all_levels:
       coeff_arr,coefs_slices = pywt.coeffs_to_array(coef,)
       plt.figure(figsize=(20,20))
       plt.imshow(coeff_arr,cmap = plt.cm.gray )
       plt.title(f'Level {Level} de wavelets')
       plt.show()

    return energy_dict

def caracteristicas_gabor(imagen):
    None