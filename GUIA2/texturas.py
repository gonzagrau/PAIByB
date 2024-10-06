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
        #'Potencia Total': mag_esp,
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
    return resultados,f_shift

def caracteristicas_wavelets(imagen, wavelet: str = 'haar', plot: bool = False, mode: str = 'periodization', Level: int = 2, Plot_all_levels: bool = False):
    # Aplicar la transformada wavelets a la imagen
    coef = pywt.wavedec2(imagen, wavelet=wavelet, level=Level, mode=mode)
    cA = coef[0]  # Coeficientes de aproximación (bajas frecuencias)
    cHVD = coef[1:]  # Coeficientes de detalles (altas frecuencias)

    # Energía de la aproximación
    energia_approx = np.sum(np.square(cA))

    # Diccionario para almacenar las energías
    energy_dict = {
        'Energía Aproximación': energia_approx
    }

    # Aplanar las energías de los detalles (horizontal, vertical, diagonal) por cada nivel
    for i, (cH, cV, cD) in enumerate(cHVD):
        energia_H = np.sum(np.square(cH))  # Energía del detalle horizontal
        energia_V = np.sum(np.square(cV))  # Energía del detalle vertical
        energia_D = np.sum(np.square(cD))  # Energía del detalle diagonal

        # Agregar la energía por nivel al diccionario de forma aplanada
        energy_dict[f'Nivel {i+1} Energía Horizontal'] = energia_H
        energy_dict[f'Nivel {i+1} Energía Vertical'] = energia_V
        energy_dict[f'Nivel {i+1} Energía Diagonal'] = energia_D

    # Ploteo si es necesario
    if Plot_all_levels:
        coeff_arr, _ = pywt.coeffs_to_array(coef)
        plt.figure(figsize=(10, 10))
        plt.imshow(coeff_arr, cmap=plt.cm.gray)
        plt.title(f'Coeficientes de wavelets nivel {Level}')
        plt.axis('off')
        plt.show()

    return energy_dict, coef
def imagen_gabor(imagen,ksize:int = 31, sigma: float = 4.0, theta: float =0, lambd:float = 10.0, gamma: float =0.5, ps: float =0):
#     ksize (Kernel Size):

#     sigma (Gaussian Envelope Width):
#     A larger sigma makes the filter smoother and less localized. If you're working with noisy data, a larger sigma might help reduce noise. For sharp details, use a smaller sigma.
#     
#     theta (Orientation):
#     To detect features at a particular angle, choose the corresponding theta. Common values are 0 (horizontal), π/2 (vertical), or intermediate angles like π/4 (45 degrees).

#     lambd (Wavelength):
#     Choose a smaller lambd for detecting finer textures and details, and a larger lambd for coarser features. You can experiment starting with values around 10.0.

#     gamma (Aspect Ratio):
#     A gamma value less than 1 makes the filter elongated and better for detecting linear features (like edges). Use values between 0.2 to 0.5 for edge detection, and closer to 1 for more circular or uniform texture detection.

#     psi (Phase Offset):
#     psi = 0 is a good starting point, which corresponds to the sine wave. If you need a different symmetry, you can try psi = π/2.

    f_gabor = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, ps, ktype=cv2.CV_32F)
    #le paso una imagen formateada con la misma profundidad que el filtro, el -1 es que la imagen resultante tenga esa profundidad 
    filtered_image =  cv2.filter2D(imagen.astype(np.float32), -1, f_gabor)

    # lo convierto devuelta a  CV_8U
    filtered_image = cv2.convertScaleAbs(filtered_image )

    return filtered_image
def caracteristicas_gabor(image_filtered):
    m_val = np.mean(image_filtered)
    var_val = np.var(image_filtered)
    return m_val, var_val

def lista_de_paths(path_folder:str):
    # Define la carpeta base
    carpeta_base = path_folder
    lista_paths =[]
    # Recorre todos los archivos dentro de la carpeta
    for root, dirs, files in os.walk(carpeta_base):
        for file in files:
            # Obtiene el path absoluto del archivo
            path_absoluto = os.path.join(root, file)
        
            lista_paths.append(path_absoluto)
            # Obtiene el path relativo con respecto a la carpeta base
            path_relativo = os.path.relpath(path_absoluto, carpeta_base)
    return lista_paths
"""
lista_paths = lista_de_paths(path_folder= 'PIByB_4')
path = lista_paths[0]
print(f"Intentando cargar la imagen desde: {path}")
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
tresh, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
image_gabor = imagen_gabor(image,theta= np.pi/2)
tresh, image_gabor = cv2.threshold(image_gabor, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.imshow(image, cmap='gray')
plt.title('Imagen original')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(image_gabor, cmap='gray')
plt.title('Imagen gabor')
plt.axis('off')

plt.show()
"""