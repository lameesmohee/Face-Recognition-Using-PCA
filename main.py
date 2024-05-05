from PyQt5.QtWidgets import QApplication, QTabWidget, QFileDialog, QLabel, QRadioButton
from PyQt5.QtCore import Qt
from PyQt5.uic import loadUi
import pyqtgraph as pg
import cv2
import numpy as np
from FaceDetection import FaceDetector



class MainWindow(QTabWidget):
    def __init__(self, ui_file):
        super().__init__()
        loadUi(ui_file, self)
        self.full_screen = False
        self.showFullScreen()
        self.applyFaceDetection = FaceDetector(self)
        self.face_recognition_BrowseBtn.clicked.connect(self.face_recognition_browse_image)
        self.face_detection_browseBtn.clicked.connect(self.face_detection_browse_image)

        


    def face_recognition_browse_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
                                                options=options)
        if file_name:
            self.selected_image_path = file_name
            image_data = cv2.imread(file_name)
            self.display_image(self.graphicsLayout_BeforeFaceRecognition, image_data)
            

    def face_detection_browse_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
                                                options=options)
        if file_name:
            self.selected_image_path = file_name
            image_data = cv2.imread(file_name)
            self.display_image(self.graphicsLayout_BeforeFaceDetection, image_data)
            self.applyFaceDetection.detectFaces()



    def display_image(self, graph_name, image_data):
        if isinstance(image_data, np.ndarray):
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            image_data = np.rot90(image_data, -1)
            graph_name.clear()
            view_box = graph_name.addViewBox()
            image_item = pg.ImageItem(image_data)
            view_box.addItem(image_item)
            view_box.autoRange()
        else:
            print("Invalid image data provided")



    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)

    





if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow("MainWindow.ui")
    window.show()
    app.exec_()
