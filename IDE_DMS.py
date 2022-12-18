import sys
import time

from PyQt5.QtCore import Qt, QTime
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QGridLayout, QLabel,QHBoxLayout,
                             QMainWindow, QProgressBar, QPushButton, QRadioButton,
                             QSpinBox, QSplitter, QVBoxLayout, QWidget)

import cv2
import numpy as np


class DenoisingWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the user interface
        self.initUI()

        # Set up the image label and buttons
        self.image_label = QLabel(self)
        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.loadImage)
        self.save_button = QPushButton('Save Image', self)
        self.save_button.clicked.connect(self.saveImage)
        self.denoise_button = QPushButton('Denoise Image', self)
        self.denoise_button.clicked.connect(self.denoiseImage)

        # Set up the radio buttons for selecting the denoising method
        self.median_button = QRadioButton('Median', self)
        self.median_button.toggled.connect(self.updateButtons)
        self.gaussian_button = QRadioButton('Gaussian', self)
        self.gaussian_button.toggled.connect(self.updateButtons)
        self.nlm_button = QRadioButton('Non-Local Means', self)
        self.nlm_button.toggled.connect(self.updateButtons)


        # Set up the spin box for adjusting the median filter kernel size
        self.median_kernel_size_spinbox = QSpinBox(self)
        self.median_kernel_size_spinbox.setRange(1, 9)
        self.median_kernel_size_spinbox.setSingleStep(1)
        self.median_kernel_size_spinbox.setValue(3)
        self.median_kernel_size_spinbox.setVisible(False)
        self.median_kernel_size_label = QLabel('Kernel size:')
        self.median_kernel_size_label.setVisible(True)
        median_layout = QHBoxLayout()
        median_layout.addWidget(self.median_kernel_size_label)
        median_layout.addWidget(self.median_kernel_size_spinbox)


        # Set up the spin box for adjusting the Gaussian filter kernel size
        self.gaussian_kernel_size_spinbox = QSpinBox(self)
        self.gaussian_kernel_size_spinbox.setRange(1, 9)
        self.gaussian_kernel_size_spinbox.setSingleStep(1)
        self.gaussian_kernel_size_spinbox.setValue(3)
        self.gaussian_kernel_size_spinbox.setVisible(False)
        self.gaussian_kernel_size_label = QLabel('Kernel size:')
        self.gaussian_kernel_size_label.setVisible(True)
        gaussian_layout = QHBoxLayout()
        gaussian_layout.addWidget(self.gaussian_kernel_size_label)
        gaussian_layout.addWidget(self.gaussian_kernel_size_spinbox)

        # Set up the spin box for adjusting the Non-Local Means filter h parameter
        self.nlm_h_spinbox = QSpinBox(self)
        self.nlm_h_spinbox.setRange(1, 99999)
        self.nlm_h_spinbox.setSingleStep(1)
        self.nlm_h_spinbox.setValue(20)
        self.nlm_h_spinbox.setVisible(False)
        self.nlm_h_label = QLabel('h:')
        self.nlm_h_label.setVisible(True)
        nlm_layout = QHBoxLayout()
        nlm_layout.addWidget(self.nlm_h_label)
        nlm_layout.addWidget(self.nlm_h_spinbox)

         # Set up the splitter
        self.splitter = QSplitter(Qt.Horizontal, self)
        self.image_label_left = QLabel(self)
        self.image_label_right = QLabel(self)
        self.splitter.addWidget(self.image_label_left)
        self.splitter.addWidget(self.image_label_right)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(1, 1)


         # Add widgets to the layout
        layout = QGridLayout(self.central_widget)
        layout.addWidget(self.splitter, 0, 0, 1, 3)
        layout.addWidget(self.load_button, 1, 0)
        layout.addWidget(self.save_button, 1, 1)
        layout.addWidget(self.denoise_button, 1, 2)
        layout.addWidget(self.median_button, 2, 0)
        layout.addWidget(self.gaussian_button, 2, 1)
        layout.addWidget(self.nlm_button, 2, 2)
        layout.addWidget(self.median_kernel_size_label, 3, 0)
        layout.addWidget(self.gaussian_kernel_size_label,3, 1)
        layout.addWidget(self.nlm_h_label,3, 2)
        layout.addWidget(self.median_kernel_size_spinbox, 4, 0)
        layout.addWidget(self.gaussian_kernel_size_spinbox, 4, 1)
        layout.addWidget(self.nlm_h_spinbox, 4, 2)



        # Set up the progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar, 5, 0, 1, 3)

    def initUI(self):
        """Set up the user interface"""
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle('Image Denoising')
        self.setGeometry(100, 100, 600, 400)



    def updateButtons(self):
        """Update the state of the radio buttons and parameter labels"""
        if self.sender() == self.median_button:
            if self.median_button.isChecked():
                self.gaussian_button.setChecked(False)
                self.nlm_button.setChecked(False)
                self.median_kernel_size_spinbox.setVisible(True)
                self.gaussian_kernel_size_spinbox.setVisible(False)
                self.nlm_h_spinbox.setVisible(False)
                self.median_kernel_size_label.setVisible(True)
                self.gaussian_kernel_size_label.setVisible(False)
                self.nlm_h_label.setVisible(False)
        elif self.sender() == self.gaussian_button:
            if self.gaussian_button.isChecked():
                self.median_button.setChecked(False)
                self.nlm_button.setChecked(False)
                self.median_kernel_size_spinbox.setVisible(False)
                self.gaussian_kernel_size_spinbox.setVisible(True)
                self.nlm_h_spinbox.setVisible(False)
                self.median_kernel_size_label.setVisible(False)
                self.gaussian_kernel_size_label.setVisible(True)
                self.nlm_h_label.setVisible(False)
        elif self.sender() == self.nlm_button:
            if self.nlm_button.isChecked():
                self.median_button.setChecked(False)
                self.gaussian_button.setChecked(False)
                self.median_kernel_size_spinbox.setVisible(False)
                self.gaussian_kernel_size_spinbox.setVisible(False)
                self.nlm_h_spinbox.setVisible(True)
                self.median_kernel_size_label.setVisible(False)
                self.gaussian_kernel_size_label.setVisible(False)
                self.nlm_h_label.setVisible(True)

    def loadImage(self):
        """Open a file dialog to select an image and display it in the label"""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Image', '',
                                                   'Images (*.png *.xpm *.jpg *.bmp);;All Files (*)',
                                                   options=options)
        if file_name:
            # Load the image and display it in the label
            self.image = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width, channels = self.image.shape
            bytes_per_line = channels * width
            qt_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.image_label_left.setPixmap(QPixmap.fromImage(qt_image))
            # self.image_label_left.setScaledContents(True)
            self.image_label_right.clear()

    def saveImage(self):
        """Open a file dialog to select a location to save the image"""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save Image', '',
                                                   'Images (*.png *.xpm *.jpg);;All Files (*)',
                                                   options=options)
        if file_name:
            # Save the image
            cv2.imwrite(file_name,cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR))

    def denoiseImage(self):
        """Apply the selected denoising method to the image"""

        # Start the progress bar and timer
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        timer = QTime()
        timer.start()

        if self.median_button.isChecked():
            kernel_size = self.median_kernel_size_spinbox.value()
            self.image = cv2.medianBlur(self.image, kernel_size)
        elif self.gaussian_button.isChecked():
            kernel_size = self.gaussian_kernel_size_spinbox.value()
            self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        elif self.nlm_button.isChecked():
            h = self.nlm_h_spinbox.value()
            self.image = cv2.fastNlMeansDenoisingColored(self.image, None, h, h, 7, 21)

         # Update the progress bar and elapsed time
        self.progress_bar.setValue(100)
        elapsed_time = timer.elapsed() / 1000

        # Display the denoised image in the right label
        height, width, channels = self.image.shape
        bytes_per_line = channels * width
        qt_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label_right.setPixmap(QPixmap.fromImage(qt_image))

        # Display the elapsed time in the status bar
        self.statusBar().showMessage('Elapsed time: {:.2f} seconds'.format(elapsed_time))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DenoisingWindow()
    window.show()
    sys.exit(app.exec_())