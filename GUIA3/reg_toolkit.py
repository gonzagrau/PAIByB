import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import fft_analysis as ffta
import pandas as pd
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
