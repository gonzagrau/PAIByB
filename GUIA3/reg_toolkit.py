import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import io, transform, img_as_float
from scipy.optimize import differential_evolution
from skimage.metrics import structural_similarity as ssim


def getPathfiles(string_path):
    list_files = os.listdir(string_path)

    # Agrego este artilugio para que ordene los archivos según el largo de su nombre
    key_ord = lambda x: int(x.split('/')[-1]\
                            .split('.')[0]\
                            .split('img-')[-1])
    list_files.sort(key=key_ord, reverse=False)

    list_pathfiles = []
    for i in list_files:
        filepath = string_path + '/' + i
        list_pathfiles.append(filepath)

    dictionary = dict()

    for i in range(len(list_files)):
        dictionary[list_files[i]] = list_pathfiles[i]

    return dictionary


def getImagesFromPathfile(list_pathfile, mode=cv2.COLOR_BGR2GRAY):
    list_imgs = []
    for i in range(len(list_pathfile)):
        img = cv2.imread(list_pathfile[i])
        img = cv2.cvtColor(img, mode)
        list_imgs.append(img)

    return list_imgs


def plotImgs(list_names, list_imgs):
    fig = plt.figure(figsize=(20, 10))

    for i in range(len(list_names)):
        ax = fig.add_subplot(1, len(list_names), i + 1)
        ax.imshow(list_imgs[i], cmap='gray')
        ax.set_title(list_names[i])
        ax.axis('off')

    plt.show()


def matchImg(fixed_img, template, a=0.1, mode=0, flip_template=False, resize_template=False, cut_template=False,
             counterclockwise=True, b=0.4):
    """
    1) mode = 0: shifting
    2) mode = 1: rotation + shifting
    3) mode = 2: trim + rotation + shifting

    """

    if flip_template:
        template = cv2.flip(template, 0)

    if resize_template:
        template = cv2.resize(template, np.shape(fixed_img))

    if cut_template:
        template = template[:fixed_img.shape[0], :fixed_img.shape[1]]

    # img.shape = (alto,ancho) = (h,w)

    h0, w0 = template.shape

    template_cut = template[int(a * h0):int((1 - a) * h0), int(a * w0):int((1 - a) * w0)]

    h1, w1 = template_cut.shape

    if mode == 0:

        # Se averigua la coordenada de máxima correlación
        res = cv2.matchTemplate(fixed_img, template_cut, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # Se traslada el template según la coordenada de máxima correlación
        M_tras_template = np.float32([[1, 0, int(max_loc[0] - a * w0)], [0, 1, int(max_loc[1] - a * h0)]])

        template_shifted = cv2.warpAffine(template, M_tras_template, fixed_img.shape[::-1])

        data = [max_val]

        return template_shifted, data

    elif mode == 1:

        if counterclockwise:
            angles = np.arange(360)
        else:
            angles = (-1) * np.arange(360)

        xcorr_list = []
        coordinates_list = []

        for angle in angles:

            if angle == 0:
                res = cv2.matchTemplate(fixed_img, template, cv2.TM_CCORR_NORMED)

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            else:
                M_rot = cv2.getRotationMatrix2D((h1 // 2, w1 // 2), angle, 1)  # Centro, ángulo, escala=1

                # Rotar la imagen a registrar
                img_rot = cv2.warpAffine(template_cut, M_rot, template_cut.shape[::-1], flags=cv2.INTER_LINEAR)

                res = cv2.matchTemplate(fixed_img, img_rot, cv2.TM_CCORR_NORMED)

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            xcorr_list.append(max_val)
            coordinates_list.append(max_loc)

        # Se averigua el ángulo y coordenada de máxima correlación

        indexes = np.arange(len(angles))

        max_xcorr = np.max(xcorr_list)

        max_xcorr_index = indexes[xcorr_list == max_xcorr][0]

        angulo_max_xcorr = angles[max_xcorr_index]

        coor_max_xcorr = coordinates_list[max_xcorr_index]

        data = [angulo_max_xcorr, max_xcorr]

        # Se rota el template al ángulo máximo de correlación
        M_rot_template = cv2.getRotationMatrix2D((h1 // 2 + int(a * h0), w1 // 2 + int(a * w0)), angulo_max_xcorr,
                                                 1)  # Centro, ángulo, escala=1
        template_rot = cv2.warpAffine(template, M_rot_template, (
            int(np.sqrt(np.sum(np.array(template.shape) ** 2))), int(np.sqrt(np.sum(np.array(template.shape) ** 2)))),
                                      flags=cv2.INTER_LINEAR)

        # Se traslada el template rotado según la coordenada de máxima correlación
        M_tras_template = np.float32([[1, 0, int(coor_max_xcorr[0] - a * w0)], [0, 1, int(coor_max_xcorr[1] - a * h0)]])

        template_rot_n_shifted = cv2.warpAffine(template_rot, M_tras_template, fixed_img.shape[::-1])

        return template_rot_n_shifted, data

    if mode == 2:
        # 1- Se toma una porción pequeña de la imagen original y se la registra en modo 1 para calcular el ángulo de rotación
        img_piece = template[int(h0 // 2 - 0.2 * h0):int(h0 // 2 + 0.2 * h0),
                    int(w0 // 2 - 0.2 * w0):int(w0 // 2 + 0.2 * w0)]

        reg_piece, data_piece = matchImg(fixed_img, img_piece, mode=1, a=0, counterclockwise=counterclockwise)
        angle_to_rotate = data_piece[0]

        # 2- Una vez calculado el ángulo, se toma ese ángulo y se rota la imagen original
        matriz = cv2.getRotationMatrix2D((h0 // 2, w0 // 2), angle_to_rotate, 1)  # Centro, ángulo, escala=1

        # tmpt = cv2.warpAffine(template, matriz, template.shape[::-1], flags=cv2.INTER_LINEAR)

        tmpt = cv2.warpAffine(template, matriz, (
            int(np.sqrt(np.sum(np.array(template.shape) ** 2))), int(np.sqrt(np.sum(np.array(template.shape) ** 2)))),
                              flags=cv2.INTER_LINEAR)

        # 3- Se supone que la imagen no necesita rotar mas, entonces se debe trasladar al lugar que coincida con la imagen fija
        reg, data = matchImg(fixed_img, tmpt, mode=0, a=b)

        return reg, data


def plotMatchComparison(fixed_img, template, registration, diff_coeff=0.8, fixed_img_name='name', template_name='name'):
    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(fixed_img, cmap='gray')
    ax.set_title(f'fixed_img - {fixed_img_name}')
    ax.axis('off')

    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(template, cmap='gray')
    ax.set_title(f'template - {template_name}')
    ax.axis('off')

    ax = fig.add_subplot(1, 4, 3)
    ax.imshow(registration, cmap='gray')
    ax.set_title(f'registration - {template_name}')
    ax.axis('off')

    ax = fig.add_subplot(1, 4, 4)
    ax.imshow(cv2.absdiff(fixed_img, np.uint8(diff_coeff * registration)), cmap='gray')
    ax.set_title('Diferencia')
    ax.axis('off')

    plt.show()


def peak_SNR(img_og: np.ndarray, img_noise: np.ndarray) -> float:
    """
    Peak signal to noise ratio estimation
    :param img_og: original image
    :param img_noise: noisy image
    :return: peak signal to noise ratio estimation
    """
    assert img_og.shape == img_noise.shape
    L = np.iinfo(img_og.dtype).max
    MSE = np.mean((img_og - img_noise) ** 2)
    return 10 * np.log(L ** 2 / MSE)


def getMetrics4Group(fixed_img, registration_imgs, registered_imgs_names):
    metrics_dic = {}
    for name, registration in zip(registered_imgs_names, registration_imgs):
        PSNR = peak_SNR(fixed_img, registration)
        SSIM = ssim(fixed_img, registration)

        metrics_dic[name] = (PSNR, SSIM)

    metrics_df = pd.DataFrame(metrics_dic).T.rename({0: 'PSNR', 1: 'SSIM'}, axis='columns')

    return metrics_df


def registracion_IM(img_ref: np.ndarray, img_mov: np.ndarray) -> np.ndarray:
    """
    Registración de imágenes utilizando la Información Mutua (MI)
    :param img_ref: imagen fija
    :param img_mov: imagen en movimiento
    :return: registracion
    """
    # Convertir las imagenes a punto flotante
    img_ref = img_as_float(img_ref)
    img_mov = img_as_float(img_mov)

    # Opcional: Redimensionar las imágenes si son demasiado grandes para acelerar la optimización
    # Puedes descomentar las siguientes líneas si es necesario
    # img_ref = transform.resize(img_ref, (256, 256), anti_aliasing=True)
    # img_mov = transform.resize(img_mov, (256, 256), anti_aliasing=True)

    # ---------------------------------------------
    # Paso 2: Implementación de la Información Mutua (MI)
    # ---------------------------------------------

    # Definir el número de bins para los histogramas conjuntos
    bins = 256

    # Calcular el histograma conjunto de dos imágenes
    hgram, x_edges, y_edges = np.histogram2d(img_ref.ravel(), img_mov.ravel(), bins=bins)

    # Convertir el histograma conjunto a probabilidades
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # Marginal de img_ref
    py = np.sum(pxy, axis=0)  # Marginal de img_mov

    # Evitar log(0) estableciendo valores mínimos
    px_py = px[:, None] * py[None, :]
    non_zero = pxy > 0  # Ignorar entradas cero

    # Calcular la Información Mutua inicial
    mi_initial = np.sum(pxy[non_zero] * np.log(pxy[non_zero] / px_py[non_zero]))
    print(f"Información Mutua inicial: {mi_initial:.6f}")

    # ---------------------------------------------
    # Paso 3: Definir la Función de Coste para la Optimización
    # ---------------------------------------------

    # Inicializar parámetros de la transformación afín
    # [a, b, c, d, e, f] donde la matriz es:
    # | a  b  e |
    # | c  d  f |
    # | 0  0  1 |
    initial_params = [1, 0, 0, 1, 0, 0]  # Transformación identidad

    # Definir la función de coste basada en Información Mutua
    # Esta función será llamada por el optimizador
    def cost_function(params):
        a, b, c, d, e, f = params
        # Construir la matriz de transformación afín
        tform_matrix = np.array([[a, b, e],
                                 [c, d, f],
                                 [0, 0, 1]])

        # Crear el objeto AffineTransform
        tform = transform.AffineTransform(matrix=tform_matrix)

        # Aplicar la transformación inversa a la imagen movida
        img_mov_transformed = transform.warp(img_mov, tform.inverse, order=3)  # Interpolación cúbica

        # Calcular el histograma conjunto entre la imagen de referencia y la imagen transformada
        hgram_transformed, _, _ = np.histogram2d(img_ref.ravel(), img_mov_transformed.ravel(), bins=bins)

        # Convertir el histograma conjunto a probabilidades
        pxy_transformed = hgram_transformed / float(np.sum(hgram_transformed))
        px_transformed = np.sum(pxy_transformed, axis=1)  # Marginal de img_ref
        py_transformed = np.sum(pxy_transformed, axis=0)  # Marginal de img_mov_transformed

        # Evitar log(0) estableciendo valores mínimos
        px_py_transformed = px_transformed[:, None] * py_transformed[None, :]
        non_zero_transformed = pxy_transformed > 0  # Ignorar entradas cero

        # Calcular la Información Mutua
        mi_transformed = np.sum(pxy_transformed[non_zero_transformed] * np.log(pxy_transformed[non_zero_transformed] / px_py_transformed[non_zero_transformed]))

        # Retornar el negativo de la MI porque 'minimize' busca minimizar
        return -mi_transformed

    # ---------------------------------------------
    # Paso 4: Optimización de los Parámetros Afines
    # ---------------------------------------------

    # Definir los límites para los parámetros de la transformación afín
    # Permitimos rotaciones hasta aproximadamente +/-45 grados y traslaciones hasta +/-20 píxeles
    bounds_affine = [
        (0.1, 1.0),    # a: escalado en X
        (-0.7, 0.7),   # b: rotación/cizallamiento en X
        (-0.7, 0.7),   # c: rotación/cizallamiento en Y
        (0.9, 1.0),    # d: escalado en Y
        (-20, 20),     # e: traslación en X
        (-20, 20)      # f: traslación en Y
    ]

    # Configurar opciones de optimización para permitir más iteraciones y mostrar progreso
    opt_options = {
        'maxiter': 1000,
        'disp': True  # Mostrar información de la optimización
    }

    # Realizar la optimización utilizando differential_evolution para una búsqueda global
    result = differential_evolution(
        cost_function,
        bounds_affine,
        maxiter=1000,
        disp=True
    )

    # Verificar si la optimización fue exitosa
    if not result.success:
        raise ValueError("La optimización no convergió: " + result.message)

    # Extraer los parámetros óptimos
    optimal_params = result.x
    print(f"Parámetros óptimos: a={optimal_params[0]:.6f}, b={optimal_params[1]:.6f}, c={optimal_params[2]:.6f}, d={optimal_params[3]:.6f}, e={optimal_params[4]:.2f}, f={optimal_params[5]:.2f}")

    # Construir la matriz de transformación afín óptima
    optimal_tform_matrix = np.array([[optimal_params[0], optimal_params[1], optimal_params[4]],
                                     [optimal_params[2], optimal_params[3], optimal_params[5]],
                                     [0,                  0,                 1          ]])

    # Crear el objeto AffineTransform con la matriz óptima
    optimal_tform = transform.AffineTransform(matrix=optimal_tform_matrix)

    # Aplicar la transformación óptima a la imagen movida
    aligned_img = transform.warp(img_mov, optimal_tform.inverse, order=3)  # Interpolación cúbica

    return aligned_img


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
