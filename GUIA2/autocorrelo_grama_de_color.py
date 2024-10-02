import cv2
import os

path = r'PAIByB\GUIA2\PIByB_3\image-1.tif'
image = cv2.imread(path)

if image is None:
    print("Error: no se pudo leer la imagen")
else:
    name = os.path.basename(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Imagen', image_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
