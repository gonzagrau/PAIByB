""" 2- Analice y compare las imágenes img-1, img-2 e img-5contenidas en la carpeta: “PAIByB-4”
I. Elija un contorno a analizar y compara los descriptores de forma de fourier y los momentos
Hu. ¿Qué puede concluir de esta comparación?¿Cual de los métodos le resulto mejor para la
extracción de características?
II. Analice las texturas del contorno elegido mediante filtros Gabor, transformada de Fuorier y
transformada wavelet. ¿Puede caracterizar de forma única la textura de los objeto mediante
los métodos utilizados?¿Qué puede decir de los valores obtenidos para cada imagen? """

import cv2
import os

import numpy as np
import matplotlib.pyplot as plt


def Contornos( image):
   #busco los maximos y los minimos valores de las intesiddes
    max = int(np.max(image))
    min = int(np.min(image))
    print(f'valor maximo de intensidad {max}')
    print(f'valor maximo de intensidad {min}')

    #calculo el umbral superior del filtro de canny mediante el metodo de otsu
    upper_tresh, _ = cv2.threshold(image, min, max, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #establesco el borde inferior proporcional al mayor 
    lower_tresh = upper_tresh*0.5

    print(f'el upper es {upper_tresh}')
    #upper_tresh = int(upper_tresh)
    #lower_tresh = int(lower_tresh)

    #calculo de los bordes por el metod de canny
    bordes = cv2.Canny(image,lower_tresh,  upper_tresh,apertureSize = 3)

    # Mostrar los resultados
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Imagen original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(bordes, cmap='gray')
    plt.title('Bordes con Canny y Otsu')
    plt.axis('off')
    plt.show()

    # encuentro los contornos uniendo los bordes 
    contornos, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos sobre la imagen original
    #imagen_contornos = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)

    imagen_contornos = bordes
    # Mostrar los resultados
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(imagen_contornos, cv2.COLOR_BGR2RGB))
    plt.title('Contornos detectados')
    plt.axis('off')
    plt.show()






#leo la imagen 
#path = r'PIByB_4/img-5.tif'
# Verificar si la imagen se cargó correctamente
# Ruta de la imagen
path = r'C:/Users/Usuario/Desktop/PIByB/PAIByB/GUIA2/PIByB_4/img-5.tif'
print(f"Intentando cargar la imagen desde: {path}")
image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el formato del archivo.")
else:
    print("Imagen cargada correctamente.")
    Contornos(image)