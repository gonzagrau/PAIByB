import cv2
from registration import Imagen, Registracion
from GUIA3.reg_toolkit import lista_de_paths, agrupar_paths
from reg_toolkit import peak_SNR, getPathfiles, matchImg


def main():
    base_dir = 'PAIByB-5'
    #extrigo paths
    lista_paths = list(getPathfiles(base_dir).values())
    # Agrupar los paths
    un_digito, dos_digitos = agrupar_paths(lista_paths)
    #aplico la clase Imagen, esta me lee la imagen y calcula los puntos calve
    un_digito_imgs = [Imagen(image) for image in un_digito ]
    dos_digito_imgs = [Imagen(image) for image in dos_digitos ]

    fixed_img = un_digito_imgs[0]
    for img in un_digito_imgs[1:]:
        print('\n' + 80 * '_')
        print(f'Imagen {img.nombre} vs. {fixed_img.nombre}')

        try:
            matchImg(img.imagen, fixed_img.imagen)
            print('match correcto con la de luki')

        except Exception as e:
            print(e)
            continue

        # Crea una instancia de Registracion utilizando Harris como descriptor
        prueba = Registracion(imagen_referencia=fixed_img, imagen_movil=img)
        prueba.registracion_CCN(mode=1)
        prueba.plot_registracion()


if __name__ == '__main__':
    main()