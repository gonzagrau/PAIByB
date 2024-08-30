import os
import sys
import cv2
import numpy as np
import pandas as pd
import json
from PyQt6 import QtWidgets
import pyqtgraph as pg
from pygments.lexer import default
from pyqtgraph import ImageView, LineSegmentROI
from pyqtgraph import RectROI


class ImageSliceAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Image Slice Analyzer')
        self.resize(1000, 1000)

        # Main widget and layout
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)

        # Widget to select roi type
        self.line_roi_radio = QtWidgets.QRadioButton('Line ROI')
        self.line_roi_radio.setChecked(True)
        self.rect_roi_radio = QtWidgets.QRadioButton('Rect ROI')

        # Button to select an image file
        self.select_button = QtWidgets.QPushButton("Select Image")
        self.select_button.clicked.connect(self.open_file_dialog)

        # Image view for displaying the image
        self.image_view = ImageView()
        self.image_view.ui.histogram.hide()
        self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()

        # Plot widget for displaying the intensity profile
        self.plot_widget = pg.PlotWidget(title="Intensity Profile")

        # Load a sample image (for demonstration purposes, we'll create a synthetic image)
        self.impath = 'PAIByB-1/Noise-1.tif'
        self.image = cv2.imread(self.impath,cv2.IMREAD_GRAYSCALE)  # Synthetic image (grayscale)

        # Display the image
        self.image_view.setImage(self.image)

        # Initialize Line ROI as default
        self.roi = LineSegmentROI([[100, 100], [400, 400]], pen='r')
        self.image_view.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_intensity_profile)

        # Connect radio buttons to toggle ROI types
        self.rect_roi_radio.toggled.connect(self.toggle_roi)
        self.line_roi_radio.toggled.connect(self.toggle_roi)

        # Save data confirm
        self.confirm_save = QtWidgets.QLabel(text='')

        # Boton de guardado
        self.save_button = QtWidgets.QPushButton(text='Save data', parent=self)
        self.save_button.clicked.connect(self.saveBasicImgInfo)

        # LAYOUTS
        self.layout = QtWidgets.QGridLayout(self.main_widget)
        self.layout.addWidget(self.line_roi_radio, 0, 0, 1, 1)
        self.layout.addWidget(self.rect_roi_radio, 0, 1, 1, 1)
        self.layout.addWidget(self.select_button, 0, 2, 1, 1)
        self.layout.addWidget(self.image_view, 1, 0, 1, 3)
        self.layout.addWidget(self.plot_widget, 2, 0, 1, 3)
        self.layout.addWidget(self.confirm_save, 3, 0, 1, 2)
        self.layout.addWidget(self.save_button, 3, 1, 1, 1)
 
    def open_file_dialog(self):
        # Set the initial directory and file filter
        # options = QtWidgets.QFileDialog.Option()\
        file_filter = "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tif)"

        # Show the file dialog and get the selected file path
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select an Image", "PAIByB-1", file_filter)
        self.update_intensity_profile()

        # Check if a file was selected
        if file_path:
            self.impath = file_path
            self.display_image()

    def toggle_roi(self):
        # Remove existing ROI before adding a new one
        self.image_view.removeItem(self.roi)

        # Create new ROI based on the selected radio button
        if self.rect_roi_radio.isChecked():
            self.roi = RectROI(pos=[100, 100], size=[50, 50], pen='r')
        elif self.line_roi_radio.isChecked():
            self.roi = LineSegmentROI([[100, 100], [400, 400]], pen='r')

        # Add the new ROI to the image view and connect its signal
        self.image_view.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.update_intensity_profile)
        self.update_intensity_profile()

    def display_image(self):
        self.image = cv2.imread(self.impath, cv2.IMREAD_GRAYSCALE)
        if self.image is not None:
            self.image_view.setImage(self.image)

    def update_intensity_profile(self):
        """
        Extract the grayscale intensity values along the ROI and plot them.
        """
        # Get the coordinates of the line
        roidata = self.roi.getArrayRegion(self.image, self.image_view.imageItem)
        
        # If no valid data is found, do nothing
        if roidata is None:
            return
        
        if self.rect_roi_radio.isChecked():
            profile = roidata.mean(axis=1)
            distances = np.arange(roidata.shape[0])
            self.plot_widget.getPlotItem().setTitle('Intensity Profile (row averaged)')

        elif self.line_roi_radio.isChecked():
            profile = roidata
            distances = np.arange(roidata.shape[0])
            self.plot_widget.getPlotItem().setTitle('Intensity Profile over line')

        # Plot the intensity profile
        self.plot_widget.clear()
        self.plot_widget.plot(distances, profile, pen='b')
        
    def saveBasicImgInfo(self):
        # OBTENCIÓN DE INFORMACIÓN
        # 1- Dimensiones de 'img'
        filename = self.impath.split('/')[-1].split('.')[0]
        img = self.roi.getArrayRegion(self.image, self.image_view.imageItem)
        pos = self.roi.pos()
        x, y = int(pos.x()), int(pos.y())
        dims = np.shape(img)

        # 2- Información básica de 'img'
        hist, std, mean = self.getBasicImgInfo(img)

        # 3 - Output
        roi_type = None
        dict_imgcarac = {}
        if self.line_roi_radio.isChecked():
            roi_type = 'line'
            dict_imgcarac = {"nombre_img": filename,
                             "x": x,
                             "y": y,
                             "roi_type": 'line',
                             "length": dims[0],
                             "desv_est": std,
                             "mean": mean}

        elif self.rect_roi_radio.isChecked():
            roi_type = 'rect'
            height = dims[0]
            width = dims[1]
            dict_imgcarac = {"nombre_img": filename,
                             "x": x,
                             "y": y,
                             "roi_type": 'rect',
                             "height": height,
                             "width": width,
                             "desv_est": std,
                             "mean": mean}

        name = f"{filename}_{x}_{y}_{roi_type}"

        # 4. Creacion de directorios

        datadir = f"infosaves/{filename}"
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        with open(f"{datadir}/{name}.json", "w") as outfile:
            json.dump(dict_imgcarac, outfile)
        
        # 2- Histograma de 'img'
        df_imghist = {"gray_lvl":      np.arange(256),
                      "hist_value":    hist}
        df_imghist = pd.DataFrame(data=df_imghist)
        df_imghist.to_csv(rf'{datadir}/hist_{name}.csv', index=None)

        self.confirm_save.setText(f'Saved data for {filename} using {roi_type} ROI selector')
    
    
    ####################################################################################
    # STATICMETHODS                                                                    #
    ####################################################################################
    
    @staticmethod
    def getBasicImgInfo(img):
        
        flat_img = np.ravel(img)
        
        # Cálculo del histograma de 'img'
        hist_img, bins1 = np.histogram(flat_img,256,[0,256])
        
        # Cálculo de la varianza de 'img'car
        std_img = np.std(flat_img)
        
        # Cálculo de la media de 'img'
        mean_img = np.mean(flat_img)
        
        return hist_img, std_img, mean_img
    
       

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ImageSliceAnalyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
