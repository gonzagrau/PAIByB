import sys
import cv2
import numpy as np
from PyQt6 import QtWidgets, QtCore
from pyqtgraph import ImageView


class HistogramRangeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SNR calculator')
        self.resize(1000, 1000)

        # Main widget and layout
        self.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)

        # Label to show the current histogram range (moved to top and larger font)
        self.range_label = QtWidgets.QLabel("Range: N/A")
        self.range_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        font = self.range_label.font()
        font.setBold(True)
        font.setPointSize(24)  # Set the font size to make the label larger
        self.range_label.setFont(font)
        self.layout.addWidget(self.range_label)

        # ImageView for displaying the image with an integrated histogram
        self.image_view = ImageView(levelMode='mono')
        self.image = cv2.imread('PAIByB-2/Pie2-4.tif', cv2.IMREAD_GRAYSCALE)
        self.image_view.setImage(self.image)

        # Histogram
        self.hist = self.image_view.getHistogramWidget()
        self.max_gray_val = np.iinfo(self.image.dtype).max
        self.bins = np.arange(self.max_gray_val)
        self.hist.setHistogramRange(0, 255)
        self.hist.orientation = 'horizontal'
        self.hist.sigLevelsChanged.connect(self.update_histogram_range)
        self.layout.addWidget(self.image_view)

    def update_histogram_range(self):
        """
        This method is called whenever the histogram range is changed.
        It updates the label to reflect the current range.
        """
        # Get the current levels from the histogram
        levels = self.hist.getLevels()
        left, right = int(levels[0]), int(levels[1])
        flat_img = self.image.flatten()
        counts, bin_edges = np.histogram(flat_img, bins=self.bins)
        bins = np.arange(len(bin_edges))
        signal_idx = bins[left: right-1]
        noise_idx = np.concat([bins[:left], bins[:right]])

        # Find SNR
        sig_avg = np.dot(counts[signal_idx], bins[signal_idx]) / counts[signal_idx].sum()
        noise_avg = np.dot(counts[noise_idx], bins[noise_idx]) / counts[noise_idx].sum()
        noise_var = np.dot((bins[noise_idx] - noise_avg)**2, counts[noise_idx]) / counts[noise_idx].sum()
        noise_sigma = np.sqrt(noise_var)
        signal_noise_ratio = (sig_avg - noise_avg) / noise_sigma

        # Update the label with the new range
        self.range_label.setText(f"SNR = {signal_noise_ratio:.2f}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = HistogramRangeApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
