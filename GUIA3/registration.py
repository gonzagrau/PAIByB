    # 1- Registre las imágenes contenidas en las carpetas 
    # I. Realice una registración basada en características evaluando los resultados con las métricas 
    # de la GUIA I de la imágenes contenidas en “PAIByB-5”¿Qué conclusiones puede sacar al 
    # respecto? 
    # II. Realice una registración basada en características evaluando los resultados con las métricas 
    # de la GUIA I de la imagenes contenidas en “PAIByB-6”¿Qué conclusiones puede sacar al 
    # respecto? 
    # III. Realice una registración basada en características evaluando los resultados con las métricas 
    # de la GUIA I de la imagenes contenidas en “PAIByB-6” previamente pre-procesadas de la 
    # manera que crea conveniente ¿Qué conclusiones puede sacar al respecto? 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from skimage import io, transform, img_as_float
from scipy import fftpack
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from skimage.metrics import normalized_mutual_information as nmi
from skimage.metrics import structural_similarity as ssim, mean_squared_error as mse
from reg_toolkit import peak_SNR as PSNR, matchImg

class Imagen:
    def __init__(self, image_path, feature_extractor='sift'):
        # Atributo: nombre de la imagen (extraído del path)
        self.nombre = image_path.split('.')[0]  # Sin la extensión del archivo
        
        # Atributo: imagen leída con OpenCV
        self.imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Verificar si la imagen se cargó correctamente
        if self.imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        if feature_extractor == 'sift':
            # Inicializar SIFT
            self.sift = cv2.SIFT_create()

            # Atributos: puntos clave (keypoints) y descriptores
            self.puntos_clave, self.descriptores = self._extraer_puntos_clave_sift()
        elif feature_extractor == 'orb':
            # Inicializar ORB
            self.orb = cv2.ORB_create()

            # Atributos: puntos clave (keypoints) y descriptores
            self.puntos_clave, self.descriptores = self._extraer_puntos_clave_orb()
        elif feature_extractor == 'harris':
            self.puntos_clave, self.descriptores = self._extraer_puntos_clave_harris()


    def _extraer_puntos_clave_sift(self):
        """
        (privado) -  extraer puntos clave y descriptores usando SIFT.
        """
        kp, descriptores = self.sift.detectAndCompute(self.imagen, mask = None)
        return kp, descriptores

    def _extraer_puntos_clave_orb(self):
        """
        (privado) -  extraer puntos clave y descriptores usando ORB.
        """
        kp = self.orb.detect(self.imagen, None)
        kp, descriptores = self.orb.compute(self.imagen, kp)
        return kp, descriptores

    def _extraer_puntos_clave_harris(self):
        """
        (privado) - extraer puntos clave y descriptores usando detección de esquinas
                    mediante el algoritmo de Harris
        """
        # Detect corners
        dst = cv2.cornerHarris(self.imagen, blockSize=2, ksize=3, k=0.04)

        # Dilate corner image to enhance corner points
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        threshold = 0.1 * dst.max()
        corner_coords = np.argwhere(dst > threshold)

        # Create keypoints
        kp = [cv2.KeyPoint(x=float(coord[1]), y=float(coord[0]), size=1) for coord in corner_coords]

        # Extract descriptors
        descriptores = np.array([dst[coord[0], coord[1]] for coord in corner_coords])

        return kp, descriptores

    def mostrar_puntos_clave(self):
        """
        (publico) - mostrar la imagen con los puntos clave detectados.
        """
        imagen_con_kp = cv2.drawKeypoints(self.imagen, self.puntos_clave, None)
        cv2.imshow('Puntos clave', imagen_con_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def info(self):
        """
        (publico) - imprimir la información de la imagen.
        """
        print(f"Nombre de la imagen: {self.nombre}")
        print(f"Número de puntos clave detectados: {len(self.puntos_clave)}")
        print(f"Descriptores: {self.descriptores.shape if self.descriptores is not None else 'No descriptores'}")

class Registracion:
    def __init__(self,
                 imagen_referencia: Imagen,
                 imagen_movil: Imagen,
                 modo: str = 'features',
                 lowe_threshold: float = 0.7,
                 min_match_count=4,
                 ransac_thres=5.0):

        assert modo in ['features', 'intensity'], "El modo debe ser 'features' o 'intensity'."
        # Heredar las imágenes de la clase ImagenSIFT
        self.img_ref = imagen_referencia
        self.img_mov = imagen_movil
        self.modo = modo

        # Definir los parámetros del emparejador FLANN
        FLANN_INDEX_KDTREE = 1
        i_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        b_params = dict(checks=50)

        # Inicializar el matcher FLANN
        self.flann = cv2.FlannBasedMatcher(i_params, b_params)

        # Atributos: matches, homografía y máscara
        self.lowe_threshold = lowe_threshold
        self.ransac_thres = ransac_thres
        self.min_match_count = min_match_count
        self.matches = None
        self.matches_filtrados = None
        self.homografia = None
        self.mask = None
        self.imagen_registrada = None  
        self.coincidencias = None 

    def calcular_matches(self):
        """
        (publico) - calcular los matches entre la imagen móvil y la imagen de referencia.
        """
        # Realizar el emparejamiento utilizando K-NN con k=2
        matches = self.flann.knnMatch(self.img_mov.descriptores, self.img_ref.descriptores, k=2)

        # Filtrar los matches usando la técnica de Lowe (ratio test)
        self.matches_filtrados = []
        for m, n in matches:
            if m.distance < self.lowe_threshold * n.distance:
                self.matches_filtrados.append(m)

        # self.matches_filtrados = matches

        self.matches = matches  # Guardar los matches originales

        print(f"Matches encontrados: {len(self.matches_filtrados)}")
    
    def dibujar_matches(self):
        """
        Para generar una imagen que representa los matches de puntos calve filtrados
        """
        self.coincidencias = cv2.drawMatches(self.img_mov.imagen,
                                             self.img_mov.puntos_clave,
                                             self.img_ref.imagen,
                                             self.img_ref.puntos_clave,
                                             self.matches_filtrados,
                                             None,
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    def calcular_homografia(self):
        """
        (publico) - calcular la homografía utilizando RANSAC.
        """
        if len(self.matches_filtrados) > self.min_match_count:
            # Extraer las coordenadas de los puntos clave coincidentes
            p_ref = np.float32([self.img_ref.puntos_clave[m.trainIdx].pt for m in self.matches_filtrados]).reshape(-1, 1, 2)
            p_movil = np.float32([self.img_mov.puntos_clave[m.queryIdx].pt for m in self.matches_filtrados]).reshape(-1, 1, 2)

            # Calcular la matriz de homografía utilizando RANSAC para eliminar outliers
            self.homografia, self.mask = cv2.findHomography(p_movil, p_ref, cv2.RANSAC, self.ransac_thres)

            #print(f"Homografía calculada:\n{self.homografia}")
        else:
            raise ValueError("No hay suficientes matches filtrados para calcular la homografía.")

    def registrar_imagen(self):
        """
        (publico) - registrar (alinear) la imagen móvil con respecto a la imagen de referencia usando la homografía.
        """
        if self.homografia is not None:
            alto, ancho = self.img_ref.imagen.shape
            self.imagen_registrada = cv2.warpPerspective(self.img_mov.imagen, self.homografia, (ancho, alto))
            return self.imagen_registrada
        else:
            raise ValueError("La homografía no ha sido calculada.")

    def plot_registration(self):
        """
        (publico) - graficar la imagen de referencia, la imagen móvil y la imagen registrada.
        """

        plt.figure(figsize=(20, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(self.img_ref.imagen, cmap='gray')
        plt.title(f'imagen ref {self.img_ref.nombre}')
        plt.axis('off')


        plt.subplot(1, 3, 2)
        plt.imshow(self.img_mov.imagen, cmap='gray')
        plt.title(f'imagen movil {self.img_mov.nombre}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(self.imagen_registrada, cmap='gray')
        plt.title('imagen registrada')
        plt.axis('off')

        plt.show()

        # Coincidencias
        if self.modo == 'features':
            plt.figure(figsize=(15, 10))
            plt.imshow(self.coincidencias,cmap= 'gray')
            plt.title(f'coincidencias {self.img_ref.nombre} con {self.img_mov.nombre}')
            plt.axis('off')
            plt.show()

        # Diferencia entre imágenes
        plt.figure(figsize=(10,10))
        diferencia = cv2.absdiff(self.img_ref.imagen, self.imagen_registrada)
        plt.imshow(diferencia, cmap='gray')
        plt.title(f'Diferencia entre Imágenes para la registración de {self.img_mov.nombre}')
        plt.axis('off')

        plt.show()

        SSIM = ssim(self.img_ref.imagen,self.imagen_registrada)
        Psnr = PSNR(self.img_ref.imagen,self.imagen_registrada)

        print(f'El valor de SSIM es {SSIM} y el de PSNR es {Psnr}')

        return SSIM, Psnr

    def run_pipeline(self, plot=True):
        """
        (publico) - ejecutar el pipeline de registración.
        """
        if self.modo == 'features':
            self.calcular_matches()
            if plot:
                self.dibujar_matches()
            self.calcular_homografia()
            self.registrar_imagen()

        elif self.modo == 'intensity':
            self.intensity_registration()

        if plot:
            self.plot_registration()

    def intensity_registration(self,
                                a=0.1,
                                mode=2,
                                flip_template=False,
                                resize_template=False,
                                cut_template=False,
                                counterclockwise=True,
                                b=0.4):
        """
        Perform intensity-based registration on the reference and moving images.
        1) mode = 0: shifting
        2) mode = 1: rotation + shifting
        3) mode = 2: trim + rotation + shifting
        """
        
        # For intensity-based registration, set the mode to 'intensity'
        self.modo = 'intensity' 
        
        fixed_img = self.img_mov.imagen
        template = self.img_ref.imagen

        # Perform intensity-based registration
        registered_img, _ = matchImg(fixed_img, template, a, mode, flip_template, resize_template, cut_template, counterclockwise, b)
        self.imagen_registrada = registered_img

        # Ajustamos el tamaño de la imagen registrada para que coincida con la de referencia
        self.imagen_registrada = cv2.resize(self.imagen_registrada, (self.img_ref.imagen.shape[1], self.img_ref.imagen.shape[0]))



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

def agrupar_paths(paths):
    un_digito = []
    dos_digitos = []

    # Expresión regular para extraer el número
    patron = re.compile(r'img-(\d+)\.tif')

    for path in paths:
        # Buscar el número en el nombre del archivo
        match = patron.search(path)
        if match:
            numero = match.group(1)
            if len(numero) == 1:
                un_digito.append(path)
            elif len(numero) == 2:
                dos_digitos.append(path)

    return un_digito, dos_digitos


def main():
    #extrigo paths 
    lista_paths = lista_de_paths('PAIByB-5')
    # Agrupar los paths
    un_digito, dos_digitos = agrupar_paths(lista_paths)
    #aplico mi clase a cada path 
    un_digito_imgs = [Imagen(image) for image in un_digito ]
    dos_digito_imgs = [Imagen(image) for image in dos_digitos ]
    for i in range(1,len(un_digito)):
        # Crea una instancia de Registracion
        prueba_reg = Registracion(imagen_referencia=un_digito_imgs[0], imagen_movil=un_digito_imgs[i])

        # Calcula los matches entre la imagen de referencia y la imagen móvil
        prueba_reg.calcular_matches()

        # Calcula las coincidencias
        prueba_reg.dibujar_matches()

        # Calcula la homografía basada en los matches filtrados
        prueba_reg.calcular_homografia()

        # Registra la imagen móvil con respecto a la de referencia
        prueba_reg.registrar_imagen()

        #grafico

        plt.figure(figsize=(20, 5))  # Ancho mayor para acomodar las tres imágenes


        plt.subplot(1, 4, 1)
        plt.imshow(prueba_reg.img_ref.imagen, cmap='gray')
        plt.title(f'Imagen ref {prueba_reg.img_ref.nombre}')
        plt.axis('off')


        plt.subplot(1, 4, 2)
        plt.imshow(prueba_reg.img_mov.imagen, cmap='gray')
        plt.title(f'Imeagen Movil {prueba_reg.img_mov.nombre}')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(prueba_reg.imagen_registrada, cmap='gray')
        plt.title('imagen registrada')
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(prueba_reg.coincidencias,cmap= 'gray')
        plt.title('coincidencias')
        plt.axis('off')
        # Mostrar el gráfico
        plt.show()
        
if __name__ == '__main__':
    main()
# un digito:
# 0 imagen de rodilla normal 
# 1 imagen rodilla rotada aprox +30
# 2 imagen rodilla flip en eje y
# 3 imagen rodilla rotacion 230 maso 

# dos digitos:
# 0 imagen rodilla oscura
# imagen rodilla oscura con rot 230




