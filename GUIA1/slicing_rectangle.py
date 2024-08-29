import sys
import cv2
import numpy as np
import pandas as pd
import json
from PyQt6 import QtWidgets
import pyqtgraph as pg
from pyqtgraph import ImageView
from pyqtgraph import RectROI


class ImageSliceAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Slice Analyzer')
        self.resize(1000, 1000)

        # Main widget and layout
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        # Image view for displaying the image
        self.image_view = ImageView()
        # self.image_view.ui.histogram.hide()
        # self.image_view.ui.roiBtn.hide()
        # self.image_view.ui.menuBtn.hide()

        # Plot widget for displaying the intensity profile
        self.plot_widget = pg.PlotWidget(title="Intensity Profile")

        # Load a sample image (for demonstration purposes, we'll create a synthetic image)
        self.image = cv2.imread('PAIByB-1/Noise-1.tif',cv2.IMREAD_GRAYSCALE)  # Synthetic image (grayscale)

        # Display the image
        self.image_view.setImage(self.image)

        # Add a rectangula ROI for the user to draw a line on the image
        self.rect_roi = RectROI(pos=[100, 100], size=[50, 50], pen='r')
        self.image_view.addItem(self.rect_roi)
        self.rect_roi.sigRegionChanged.connect(self.update_intensity_profile)
        
        # Boton de guardado
        self.save_button = QtWidgets.QPushButton(text='Save data', parent=self)
        self.save_button.clicked.connect(self.saveBasicImgInfo)

        # LAYOUTS
        self.layout = QtWidgets.QGridLayout(self.main_widget)
        self.layout.addWidget(self.image_view, 0, 0, 1, 2)
        self.layout.addWidget(self.plot_widget, 1, 0, 1, 2)
        self.layout.addWidget(self.save_button, 2, 1)
 
        
    def update_intensity_profile(self):
        """
        Extract the grayscale intensity values along the ROI and plot them.
        """
        # Get the coordinates of the line
        rect_data = self.rect_roi.getArrayRegion(self.image, self.image_view.imageItem)
        
        # If no valid data is found, do nothing
        if rect_data is None:
            return
        

        rect_mean = rect_data.mean(axis=0)
        # Calculate the distance along the line
        distances = np.arange(rect_data.shape[1])

        # Plot the intensity profile
        self.plot_widget.clear()
        self.plot_widget.plot(distances, rect_mean, pen='b')
        
    def saveBasicImgInfo(self):
        
        # OBTENCIÓN DE INFORMACIÓN
        # 1- Dimensiones de 'img'
        img = self.rect_roi.getArrayRegion(self.image, self.image_view.imageItem)
        pos = self.rect_roi.pos()
        x, y = pos.x(), pos.y()
        dims = np.shape(img)
        height = dims[0]
        width = dims[1]
        name = f"{x}_{y}_{height}x{width}"
        
        # 2- Información básica de 'img'
        hist, var, mean = self.getBasicImgInfo(img)
        
        # CREACIÓN DE DATAFRAMES
        # 1- Resumen de características de 'img'
        dict_imgcarac = {"nombre_img":    name,
                       "altura":        height,
                       "width":         width,
                       "varianza":      var,
                       "mean":          mean}
        with open(f"infosaves/{name}.json", "w") as outfile: 
            json.dump(dict_imgcarac, outfile)
        
        # 2- Histograma de 'img'
        df_imghist = {"gray_lvl":      np.arange(256),
                      "hist_value":    hist}
        df_imghist = pd.DataFrame(data=df_imghist)
        df_imghist.to_csv(f'infosaves/hist{name}.csv', index=None)

        print(f'Saved data for {name}')
    
    
    ####################################################################################
    # STATICMETHODS                                                                    #
    ####################################################################################
    
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
    
       

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ImageSliceAnalyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
