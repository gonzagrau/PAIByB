import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from reg_toolkit import lista_de_paths, agrupar_paths
from reg_toolkit import peak_SNR as PSNR, matchImg, registracion_IM

class Imagen:
    def __init__(self, image_path, feature_extractor='sift', harris_thres=0.1):
        # Atributo: nombre de la imagen (extraído del path)
        self.nombre = image_path.split('.')[0]  # Sin la extensión del archivo
        
        # Atributo: imagen leída con OpenCV
        self.imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Verificar si la imagen se cargó correctamente
        if self.imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")

        # Validad extractor de caracteristicas
        assert feature_extractor in {'sift', 'orb', 'harris'},\
            "El extractor de características debe ser 'sift', 'orb' o 'harris'."
        self.feature_extractor = feature_extractor
        self.extraer_puntos_clave()
        # Atributos: puntos clave y descriptores


    def show(self):
        """
        (publico) - mostrar la imagen en una ventana.
        """
        plt.imshow(self.imagen, cmap='gray', vmin=0, vmax=255)
        plt.title(self.nombre)
        plt.axis('off')
        plt.show()

    def _extraer_puntos_clave_sift(self):
        """
        (privado) -  extraer puntos clave y descriptores usando SIFT.
        """
        self.sift = cv2.SIFT_create()
        kp, descriptores = self.sift.detectAndCompute(self.imagen, mask = None)
        return kp, descriptores

    def _extraer_puntos_clave_orb(self):
        """
        (privado) -  extraer puntos clave y descriptores usando ORB.
        """
        self.orb = cv2.ORB_create()
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
        threshold = self.harris_thres * dst.max()
        corner_coords = np.argwhere(dst > threshold)

        # Create keypoints
        size = self.imagen.shape[2] if len(self.imagen.shape) == 3 else 1
        kp = [cv2.KeyPoint(x=float(coord[1]), y=float(coord[0]), size=size) for coord in corner_coords]

        # Extract descriptors  # Extract descriptors
        descriptores = np.array([[dst[coord[0], coord[1]]]/dst.max() for coord in corner_coords], dtype=np.float32)

        return kp, descriptores

    def extraer_puntos_clave(self):
        """
        (publico) - extraer los puntos clave de la imagen.
        """
        if self.feature_extractor == 'sift':
            self.puntos_clave, self.descriptores = self._extraer_puntos_clave_sift()

        elif self.feature_extractor == 'orb':
            self.puntos_clave, self.descriptores = self._extraer_puntos_clave_orb()

        elif self.feature_extractor == 'harris':
            self.puntos_clave, self.descriptores = self._extraer_puntos_clave_harris()


    def mostrar_puntos_clave(self):
        """
        (publico) - mostrar la imagen con los puntos clave detectados.
        """
        imagen_con_kp = cv2.drawKeypoints(self.imagen, self.puntos_clave, None)
        plt.imshow(imagen_con_kp)
        plt.title(f"Puntos clave detectados en {self.nombre}")
        plt.axis('off')
        plt.show()

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

        assert modo in ['features', 'intensidad'], "El modo debe ser 'features' o 'intensidad'."
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

    def plot_registracion(self):
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

    def ejecutar_registracion(self, plot=True):
        """
        (publico) - ejecutar el pipeline de registración.
        """
        if self.modo == 'features':
            self.calcular_matches()
            if plot:
                self.dibujar_matches()
            self.calcular_homografia()
            self.registrar_imagen()

        elif self.modo == 'intensidad':
            self.registracion_CCN()

        if plot:
            self.plot_registracion()
            self.calcular_metricas(verbose=True)

    def registracion_CCN(self,
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
        
        # For intensity-based registration, set the mode to 'intensidad'
        self.modo = 'intensidad'
        
        fixed_img = self.img_mov.imagen
        template = self.img_ref.imagen

        # Perform intensity-based registration
        registered_img, _ = matchImg(fixed_img, template, a, mode, flip_template, resize_template, cut_template, counterclockwise, b)
        self.imagen_registrada = registered_img

        # Ajustamos el tamaño de la imagen registrada para que coincida con la de referencia
        self.imagen_registrada = cv2.resize(self.imagen_registrada, (self.img_ref.imagen.shape[1], self.img_ref.imagen.shape[0]))

    def calcular_metricas(self, verbose=True):
        """
        SSIM, MSE y PSNR
        :return: valores numericos de SSIM, MSE y PSNR en formato DATAFRAME
        """
        SSIM_val = ssim(self.img_ref.imagen,self.imagen_registrada)
        MSE_val = np.mean((self.img_ref.imagen - self.imagen_registrada)**2)
        PSNR_val = PSNR(self.img_ref.imagen,self.imagen_registrada)

        if verbose:
            print(f"SSIM: {SSIM_val:.4f}")
            print(f"MSE: {MSE_val:.4f}")
            print(f"PSNR: {PSNR_val:.4f}")

        df = pd.DataFrame({'SSIM': [SSIM_val], 'MSE': [MSE_val], 'PSNR': [PSNR_val]})
        return df

    def registracion_IM(self):
        """
        Aplica el metodo IM definido en reg_toolkit
        :return: la imagen registrada
        """
        img_mov, img_ref, img_reg = registracion_IM(self.img_mov.imagen, self.img_ref.imagen)
        self.img_mov.imagen = img_mov.astype(self.img_mov.imagen.dtype)
        self.img_ref.imagen = img_ref.astype(self.img_mov.imagen.dtype)
        self.imagen_registrada = img_reg.astype(self.img_mov.imagen.dtype)

        return self.imagen_registrada
    
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




