import cv2
from PyQt5 import QtGui, QtWidgets

class ZoomGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Enable the rubber band drag mode
        self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)

        # Set the transformation anchor to Anchor the zoom at the center of the viewport
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event):
        # Check if the Control key is pressed
        if event.modifiers() & QtCore.Qt.ControlModifier:
            # Calculate the zoom factor
            zoom_factor = 1.1 ** event.angleDelta().y()

            # Apply the zoom transformation
            self.scale(zoom_factor, zoom_factor)
        else:
            # Call the base class implementation if the Control key is not pressed
            super().wheelEvent(event)

# Load the image using OpenCV
image = cv2.imread("examples/10081.jpg")

# Convert the image to a QImage object
qimage = QtGui.QImage(image, image.shape[1], image.shape[0], image.strides[0], QtGui.QImage.Format_RGB888)

# Create a ZoomGraphicsView widget and set the QImage as the background image
view = ZoomGraphicsView()
view.setRenderHint(QtGui.QPainter.Antialiasing)
view.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
view.setScene(QtWidgets.QGraphicsScene())
view.scene().addPixmap(QtGui.QPixmap.fromImage(qimage))

# Show the ZoomGraphicsView
view.show()