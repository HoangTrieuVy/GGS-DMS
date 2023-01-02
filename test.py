import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplitter, QGraphicsView, QGraphicsScene
from PyQt5.QtGui import QPen, QBrush, QPainterPath
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QLabel, QComboBox, 
                             QHBoxLayout, QVBoxLayout, QLineEdit, QPushButton,QSplitter,QListWidgetItem,QListWidget)
from PyQt5.QtGui import QPixmap,QPainter, QBrush, QPen, QPainterPath
from PyQt5.QtCore import Qt, QTime,QPointF
from PyQt5.QtGui import QImage
from PyQt5 import QtCore, QtGui, QtWidgets
import time
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def generate_curve_data():
    # Generate x and y values for the curve using NumPy
    x = np.linspace(-np.pi, np.pi, 10000)
    y = np.sin(x)
    return x, y

app = QApplication([])
window = QMainWindow()
splitter = QSplitter(window)

view = QGraphicsView(splitter)
scene = QGraphicsScene(view)
view.setScene(scene)

x, y = generate_curve_data()
pen = QPen(Qt.red, 2)
brush = QBrush(Qt.red)
path = QPainterPath()
path.moveTo(x[0], y[0])
for i in range(1, len(x)):
    path.lineTo(x[i], y[i])
curve = scene.addPath(path, pen, brush)

window.show()
app.exec_()