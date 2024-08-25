import sys
import cv2
import numpy as np
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
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        # Image view for displaying the image
        self.image_view = ImageView()
        self.layout.addWidget(self.image_view)

        # Plot widget for displaying the intensity profile
        self.plot_widget = pg.PlotWidget(title="Intensity Profile")
        self.layout.addWidget(self.plot_widget)

        # Load a sample image (for demonstration purposes, we'll create a synthetic image)
        self.image = cv2.imread('PAIByB-1/Noise-1.tif',cv2.IMREAD_GRAYSCALE)  # Synthetic image (grayscale)

        # Display the image
        self.image_view.setImage(self.image)

        # Add a line ROI for the user to draw a line on the image
        self.rect_roi = RectROI(pos=[100, 100], size=[50, 50], pen='r')
        
        self.image_view.addItem(self.rect_roi)

        # Connect the ROI's sigRegionChanged signal to update the plot
        self.rect_roi.sigRegionChanged.connect(self.update_intensity_profile)

    def update_intensity_profile(self):
        """
        Extract the grayscale intensity values along the line defined by the ROI and plot them.
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

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ImageSliceAnalyzer()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
