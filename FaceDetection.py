import numpy as np
import cv2
import pyqtgraph as pg

class FaceDetector:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget

    def detectFaces(self):
        if self.main_tab_widget.selected_image_path:
            imageArray = cv2.imread(self.main_tab_widget.selected_image_path)
            if imageArray.ndim == 3:
                imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            self.imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            originalImage = self.imageArray
        print('Done')
