import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QLabel, QComboBox, 
                             QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,QSplitter,QListWidgetItem,QListWidget)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtGui import QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import time
import matplotlib.pyplot as plt

import cv2
import numpy as np
sys.path.insert(0, 'python_dms/lib/')
from tools_dms import *
from tools_trof import *
from PIL import Image
import scipy as scp
import scipy.io

class DenoiserWidget(QWidget):
    def __init__(self):
        super().__init__()

        plt.figure()

        # Create a combo box to choose the denoising method
        self.method_combo_box = QComboBox()
        self.method_combo_box.addItem("medianBlur")
        self.method_combo_box.addItem("DMS")
        self.method_combo_box.addItem("TV")

        # Create a line edit to set the parameter for method 2
        self.algo_combo_box = QComboBox()
        self.algo_combo_box.insertItem(0, "Select an algorithm")
        self.algo_combo_box.addItem("PALM")
        self.algo_combo_box.addItem("SLPAM")
        self.algo_combo_box.setVisible(False)  # initially hidden
        # Set the current index to the placeholder item
        self.algo_combo_box.setCurrentIndex(0)

        self.length_penalization_combo_box = QComboBox()
        self.length_penalization_combo_box.insertItem(0, "Select a lenght penalization")
        self.length_penalization_combo_box.addItem("l1")
        self.length_penalization_combo_box.addItem("AT")
        self.length_penalization_combo_box.setVisible(False)  # initially hidden
        self.length_penalization_combo_box.setCurrentIndex(0)
        self.beta_DMS_line_edit = QLineEdit()
        self.beta_DMS_line_edit.setPlaceholderText("Beta: default using Golden-Grid-Search")
        self.beta_DMS_line_edit.setVisible(False)  # initially hidden
        self.lambda_DMS_line_edit = QLineEdit()
        self.lambda_DMS_line_edit.setPlaceholderText("Lambda: default using Golden-Grid-Search")
        self.lambda_DMS_line_edit.setVisible(False)  # initially hidden

        # Create a line edit to set the parameter for method 1
        self.param1_line_edit = QLineEdit()
        self.param1_line_edit.setPlaceholderText("Kernel size")
        self.param1_line_edit.setVisible(True)  # initially hidden




        # Create a line edit to set the parameter for method 3
        self.algo_tv_combo_box = QComboBox()
        self.algo_tv_combo_box.insertItem(0, "Select an algorithm")
        self.algo_tv_combo_box.addItem("Forward-Backward")
        self.algo_tv_combo_box.addItem("Chambolle-Pock")
        self.algo_tv_combo_box.setVisible(False)  # initially hidden
        self.algo_combo_box.setCurrentIndex(0)

        self.param3_line_edit = QLineEdit()
        self.param3_line_edit.setPlaceholderText("Lambda: ")
        self.param3_line_edit.setVisible(False)  # initially hidden
        self.regulizer_combo_box = QComboBox()
        self.regulizer_combo_box.insertItem(0, "Select a regularizer")
        self.regulizer_combo_box.addItem("l1")
        self.regulizer_combo_box.addItem("l2")
        self.regulizer_combo_box.setVisible(False)  # initially hidden
        self.regulizer_combo_box.setCurrentIndex(0)

        # Create a button to load the image
        self.load_button = QPushButton("Load image")
        self.load_button.clicked.connect(self.on_load_button_clicked)
        self.save_button = QPushButton('Save Image', self)
        self.save_button.clicked.connect(self.saveImage)

        # Create a button to denoise the image
        self.denoise_button = QPushButton("Denoise")
        self.denoise_button.clicked.connect(self.on_denoise_button_clicked)

        # Create a label to display the image size
        self.image_size_label = QLabel()
        # Create a label to display the denoising time
        self.denoising_time_label = QLabel()

        # Create a horizontal layout for the denoising controls
        denoising_controls_layout = QVBoxLayout()
        denoising_controls_layout.addWidget(self.method_combo_box)
        denoising_controls_layout.addWidget(self.param1_line_edit)
        denoising_controls_layout.addWidget(self.algo_combo_box)
        denoising_controls_layout.addWidget(self.length_penalization_combo_box)
        denoising_controls_layout.addWidget(self.beta_DMS_line_edit)
        denoising_controls_layout.addWidget(self.lambda_DMS_line_edit)
        denoising_controls_layout.addWidget(self.param3_line_edit)
        denoising_controls_layout.addWidget(self.algo_tv_combo_box)
        denoising_controls_layout.addWidget(self.regulizer_combo_box)
        denoising_controls_layout.addWidget(self.load_button)
        denoising_controls_layout.addWidget(self.save_button)
        denoising_controls_layout.addWidget(self.denoise_button)
        denoising_controls_layout.addWidget(self.image_size_label)
        denoising_controls_layout.addWidget(self.denoising_time_label)

        # Set up the splitter
        self.splitter1 = QSplitter(Qt.Horizontal, self)
        self.image_label_left = QLabel(self)
        self.image_label_right = QLabel(self)
        self.splitter2 = QSplitter(Qt.Horizontal, self)
        self.image_label_left_down = QLabel(self)        
        self.image_label_right_down = QLabel(self)
        self.splitter1.addWidget(self.image_label_left)
        self.splitter1.addWidget(self.image_label_right)
        self.splitter2.addWidget(self.image_label_left_down)
        self.splitter2.addWidget(self.image_label_right_down)
        self.splitter1.setStretchFactor(1, 1)
        self.splitter2.setStretchFactor(1, 1)

        # Create a vertical layout for the widget
        layout = QHBoxLayout()
        denoising_controls_layout.addWidget(self.splitter1)
        denoising_controls_layout.addWidget(self.splitter2)
        layout.addLayout(denoising_controls_layout)
        self.setLayout(layout)



        # Connect the method_combo_box currentIndexChanged signal to the on_method_combo_box_changed slot
        self.method_combo_box.currentIndexChanged.connect(self.on_method_combo_box_changed)

    def on_load_button_clicked(self):
        # Open a file dialog to choose an image
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            # Load the image and display it in the image_label
            # image = QPixmap(file_name)
            # self.image_label_left.setPixmap(image)

            # Load the image and display it in the label
            self.image_noisy = cv2.imread(file_name, cv2.IMREAD_COLOR)
            self.image_noisy = cv2.cvtColor(self.image_noisy, cv2.COLOR_BGR2GRAY) #cv2.COLOR_BGR2RGB)
            self.image_noisy_numpy = np.asarray( self.image_noisy, dtype="int32" )
            # height, width, channels = self.image_noisy.shape
            height, width = self.image_noisy.shape
            channels= 1
            # Set the image size label
            self.image_size_label.setText(f"Image size: {width}x{height}")

            bytes_per_line = channels * width
            qt_image = QImage(self.image_noisy.data, width, height, bytes_per_line, QImage.Format_Grayscale8)#Format_RGB888)
            self.image_label_left.setPixmap(QPixmap.fromImage(qt_image))
            self.image_label_left.setScaledContents(True)
            self.image_label_right.clear()


    def on_denoise_button_clicked(self):
        # Denoise the image using the selected method and parameters
        method = self.method_combo_box.currentText()
        # print(f"Denoising image using method {method} with params: {param1}, {param2}, {param3}")
        # Add your denoising code here

        if method == "medianBlur":
            param1 = int(self.param1_line_edit.text())
            # Denoise using method 1 with parameter param1
            denoised_image = self.denoise_method1(self.image_noisy, param1)

        elif method == "DMS":

            start = time.time()


            algo = self.algo_combo_box.currentText()
            normtype = self.length_penalization_combo_box.currentText()
            beta = float(self.beta_DMS_line_edit.text())
            lamb = float(self.lambda_DMS_line_edit.text())
            A  = np.ones_like(self.image_noisy_numpy)
            mit=300

            model = DMS(
                    norm_type=normtype,
                    edges="similar",
                    beta=beta,
                    lamb=lamb,
                    eps=0.2,
                    stop_criterion=1e-4,
                    MaximumIteration=mit,
                    method=algo,
                    noised_image_input=self.image_noisy,
                    optD="OptD",
                    dk_SLPAM_factor=1e-4,
                    eps_AT_min=0.02,
                    A=A)

            out = model.process()
            denoised_image = out[1]


        elif method == "TV":
            param3 = self.param3_line_edit.text()
            # Denoise using method 3 with parameter param1
            denoised_image = self.denoise_method3(image, param1)

        # Display the denoised image in the image_label
        self.image_label_right.setPixmap(self.to_qpixmap(np.asarray( np.clip(denoised_image*255,0,255), dtype="uint8")))
        # self.image_label_left_down.setPixmap(self.to_qpixmap(np.asarray( np.clip(out[5][:,:,0]*255,0,255), dtype="uint8")))
        
        # plt.plot(out[3],label=self.algo_combo_box.currentText()+self.length_penalization_combo_box.currentText())
        # plt.legend()
        # plt.show()
        # self.image_label_right_down.setPixmap(self.to_qpixmap(np.asarray( np.clip((out[5][:,:,1]+out[5][:,:,0])/2*255,0,255), dtype="uint8")))

        # Set the denoising time label
        end = time.time()
        self.denoising_time_label.setText(f"Denoising time: {end - start:.2f} seconds")

        self.image_denoised= np.asarray( np.clip(denoised_image*255,0,255), dtype="uint8")

    def saveImage(self):
        """Open a file dialog to select a location to save the image"""
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(self, 'Save Image', '',
                                                   'Images (*.png *.xpm *.jpg);;All Files (*)',
                                                   options=options)
        if file_name:
            # Save the image
            cv2.imwrite(file_name,cv2.cvtColor(self.image_denoised, cv2.COLOR_RGB2BGR))

    def to_qpixmap(self, image):
        # Convert the image to a QPixmap
        height, width = image.shape
        # height, width, channel = image.shape
        # bytes_per_line = 3 * width
        bytes_per_line = 1 * width
        return QPixmap.fromImage(QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)) #Format_RGB888))

    def on_method_combo_box_changed(self, index):
        # Show the appropriate parameter line edit when the method is changed
        if index == 0:
            self.param1_line_edit.setVisible(True)
            self.algo_combo_box.setVisible(False)
            self.length_penalization_combo_box.setVisible(False)
            self.beta_DMS_line_edit.setVisible(False)
            self.lambda_DMS_line_edit.setVisible(False)
            self.param3_line_edit.setVisible(False)
            self.regulizer_combo_box.setVisible(False)
            self.algo_tv_combo_box.setVisible(False)
        elif index == 1:
            self.param1_line_edit.setVisible(False)
            self.algo_combo_box.setVisible(True)
            self.length_penalization_combo_box.setVisible(True)
            self.beta_DMS_line_edit.setVisible(True)
            self.lambda_DMS_line_edit.setVisible(True)
            self.param3_line_edit.setVisible(False)
            self.regulizer_combo_box.setVisible(False)
            self.algo_tv_combo_box.setVisible(False)
        elif index == 2:
            self.param1_line_edit.setVisible(False)
            self.algo_combo_box.setVisible(False)
            self.length_penalization_combo_box.setVisible(False)
            self.beta_DMS_line_edit.setVisible(False)
            self.lambda_DMS_line_edit.setVisible(False)
            self.param3_line_edit.setVisible(True)
            self.regulizer_combo_box.setVisible(True)
            self.algo_tv_combo_box.setVisible(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = DenoiserWidget()
    widget.show()
    sys.exit(app.exec_())