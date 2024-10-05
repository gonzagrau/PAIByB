""" 2- Analice y compare las imágenes img-1, img-2 e img-5contenidas en la carpeta: “PAIByB-4”
I. Elija un contorno a analizar y compara los descriptores de forma de fourier y los momentos
Hu. ¿Qué puede concluir de esta comparación?¿Cual de los métodos le resulto mejor para la
extracción de características?
II. Analice las texturas del contorno elegido mediante filtros Gabor, transformada de Fuorier y
transformada wavelet. ¿Puede caracterizar de forma única la textura de los objeto mediante
los métodos utilizados?¿Qué puede decir de los valores obtenidos para cada imagen? """

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
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
        

def Contornos(image, otsu: bool = False, upper_tresh: int = 200, lower_tresh: int = 100, plot: bool = False):
    # Si se desea usar el umbral de Otsu
    if otsu:
        upper_tresh, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lower_tresh = upper_tresh * 0.5

    # Aplicar el filtro Canny para detectar bordes
    bordes = cv2.Canny(image, lower_tresh, upper_tresh, apertureSize=3)

    # Encontrar los contornos
    contornos, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if plot:
        # Crear una imagen negra y dibujar los contornos detectados sobre ella
        imagen_contornos = np.zeros_like(image)  # Imagen en negro
        cv2.drawContours(imagen_contornos, contornos, -1, 255, 1)  # 1 píxel de grosor, color blanco

        # Mostrar la imagen original y la imagen con los contornos
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Imagen Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(imagen_contornos, cmap='gray')
        plt.title(f'{len(contornos)} Contornos Detectados')
        plt.axis('off')

        plt.show()

    return contornos

def contorno2fourier(borde, porcentaje_descriptores: float):
    # Convertir el contorno a formato complejo
    borde_complejo = np.array([point[0][0] + 1j * point[0][1] for point in borde])
    
    # Aplicar FFT
    borde_fourier = np.fft.fft(borde_complejo)
    borde_fourier = np.fft.fftshift(borde_fourier)

    # Calcular el número de descriptores basado en el porcentaje
    n_total_frecuencias = len(borde_fourier)
    n_descriptores = int(n_total_frecuencias * porcentaje_descriptores)

    # Filtrar y mantener solo los primeros y últimos n descriptores
    fourier_filtrado = np.zeros(borde_fourier.shape, dtype=complex)
    fourier_filtrado[:n_descriptores] = borde_fourier[:n_descriptores]
    fourier_filtrado[-n_descriptores:] = borde_fourier[-n_descriptores:]

    return fourier_filtrado, borde_fourier

def reconstruir_contorno(contorno_fourier, shift: bool = True):
    # Revertir el FFT shift si fue aplicado
    if shift:
        contorno_fourier = np.fft.ifftshift(contorno_fourier)
        
    # Transformada Inversa de Fourier para reconstruir el contorno
    contorno_reconstruido = np.fft.ifft(contorno_fourier)
    
    # Convertir de complejo a coordenadas reales
    contorno_reconstruido = np.stack((contorno_reconstruido.real, contorno_reconstruido.imag), axis=-1).astype(int)
    
    return contorno_reconstruido

# Función para reconstruir todos los contornos y combinarlos en una sola imagen
def reconstruir_todos_los_contornos(image, contornos, n_descriptores,invariante: bool = False):
    # Crear una imagen vacía (negra) del mismo tamaño que la original
    imagen_reconstruida = np.zeros_like(image)

    for contorno in contornos:
        # Si el contorno tiene al menos 2 puntos (para aplicar FFT)
        if len(contorno) > 1:
            # Transformar a Fourier y reconstruir el contorno
            if invariante:
                f_contorno = contorno2fourier_invariante(contorno, n_descriptores=n_descriptores)
                contorno_reconstruido = reconstruir_contorno(f_contorno)

            else:
                f_contorno = contorno2fourier(contorno, n_descriptores=n_descriptores)
                contorno_reconstruido = reconstruir_contorno(f_contorno)

            # Filtrar puntos que están fuera de la imagen (evitar errores de índice)
            contorno_reconstruido = contorno_reconstruido[
                (contorno_reconstruido[:, 0] >= 0) & (contorno_reconstruido[:, 0] < image.shape[1]) &
                (contorno_reconstruido[:, 1] >= 0) & (contorno_reconstruido[:, 1] < image.shape[0])
            ]

            # Dibujar el contorno reconstruido sobre la imagen negra
            for punto in contorno_reconstruido:
                cv2.circle(imagen_reconstruida, (punto[0], punto[1]), 1, 255, -1)  # Dibuja un píxel blanco por cada punto

    return imagen_reconstruida
def hacer_invariante_fourier(contorno_fourier):
    # Ignorar la frecuencia cero (traslación)
    contorno_invariante = contorno_fourier[1:]

    # Normalizar la escala (dividir por el primer coeficiente)
    contorno_invariante = contorno_invariante / np.abs(contorno_invariante[0])

    # Usar solo las magnitudes (invariante a rotación)
    return np.abs(contorno_invariante)
def contorno2fourier_invariante(borde, n_descriptores: int):
    # Convertir el contorno a formato complejo
    borde_complejo = np.array([point[0][0] + 1j * point[0][1] for point in borde])
    
    # Aplicar FFT
    borde_fourier = np.fft.fft(borde_complejo)
    
    # Hacer los descriptores invariantes a traslación, rotación y escala
    fourier_invariante = hacer_invariante_fourier(borde_fourier)

    # Filtrar y mantener solo los primeros n descriptores
    fourier_filtrado = np.zeros(fourier_invariante.shape, dtype=complex)
    fourier_filtrado[:n_descriptores] = fourier_invariante[:n_descriptores]

    return fourier_filtrado

def Hu_moments(contorno, label, plot:bool = False):
    # Calcular los Momentos de Hu
    hu_moment = cv2.HuMoments(cv2.moments(contorno)).flatten()

    # Aplicar logaritmo a los momentos de Hu para su mejor visualización
    log_hu_mo = -np.sign(hu_moment) * np.log10(np.abs(hu_moment))

    # Crear un diccionario para almacenar los momentos de Hu con el label
    momentos_dict = {
        'label': label,
        'Hu_Moments': {f'Hu_Moment_{i+1}': log_hu_mo[i] for i in range(len(log_hu_mo))}
    }
    if plot:

        # Mostrar los valores de los descriptores de Hu en consola
        print(f"Label: {label}")
        print("Hu Moments:")
        for i, moment in enumerate(log_hu_mo):
            print(f"Hu Moment {i+1}: {moment}")

    return momentos_dict  # Retornar el diccionario con los momentos de Hu y el label
"""
# Leer la imagen
lista_paths = lista_de_paths(path_folder= 'PIByB_4')
path = lista_paths[0]
print(f"Intentando cargar la imagen desde: {path}")
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    print("Imagen cargada correctamente.")
    
    # Obtener contornos
    contornos = Contornos(image, upper_tresh=200, lower_tresh=100, plot=False)
    # Crear una imagen negra para mostrar los contornos originales
    imagen_contornos_originales = np.zeros_like(image)
    cv2.drawContours(imagen_contornos_originales, contornos, -1, 255, 1)  # Dibuja todos los contornos originales
    
    # Asegúrate de que se encontraron contornos
    if len(contornos) > 0:
        # Reconstruir todos los contornos y combinarlos en una imagen
        imagen_reconstruida = reconstruir_todos_los_contornos(image, contornos, n_descriptores=50)
        imagen_reconstruida_inv = reconstruir_todos_los_contornos(image, contornos, n_descriptores=50,invariante=True)

        # Mostrar los resultados
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Imagen original')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(imagen_reconstruida, cmap='gray')
        plt.title('Imagen reconstruida con Fourier')
        plt.axis('off')

        

        plt.subplot(1, 3, 3)
        plt.imshow(imagen_contornos_originales, cmap='gray')
        plt.title('contornos')
        plt.axis('off')

        plt.show()
    else:
        print("No se encontraron contornos.")
    

"""

