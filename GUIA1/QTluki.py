import sys
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QMainWindow

import numpy as np
import pyqtgraph as pg
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Carga el archivo .ui
        uic.loadUi("GUIA1/archivosUI/MainWindow_v01.ui", self)
        
        self.image_view = pg.ImageView(parent=self.gra_imgSeleccionada)
        self.image = cv2.imread("GUIA1/PAIByB-1/Noise-1.tif",cv2.IMREAD_GRAYSCALE)  # Synthetic image (grayscale)
        self.image_view.setImage(self.image)
        
        self.rect_roi = pg.RectROI(pos=[100, 100], size=[50, 50], pen='r')
        self.image_view.addItem(self.rect_roi)
        
        self.hist_view = pg.PlotWidget(parent=self.gra_histograma)
        self.hist_view.plotItem(x=np.arange(256),)
        
        self.btn_abrirImg.clicked.connect(self.click)
        
    def click(self):
        print("Lukitesting")
        

    @staticmethod
    def getBasicImgInfo(img):
        
        flat_img = np.ravel(img)
        
        # Cálculo del histograma de 'img'
        hist_img, bins1 = np.histogram(flat_img,256,[0,256])
        
        # Cálculo de la varianza de 'img'car
        var_img = np.var(flat_img)
        
        # Cálculo de la media de 'img'
        mean_img = np.mean(flat_img)
        
        return hist_img, var_img, mean_img

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

