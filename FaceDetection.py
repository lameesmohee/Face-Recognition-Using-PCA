import numpy as np
import cv2
import pyqtgraph as pg
import matplotlib.pyplot as plt

class FaceDetector:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget

    def detectFaces(self):
        if self.main_tab_widget.selected_image_path:
            imageBGR = cv2.imread(self.main_tab_widget.selected_image_path)
            imageGray = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2GRAY)
        
        faceDetector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        face = faceDetector.detectMultiScale(imageGray, scaleFactor=1.1, minNeighbors=20, minSize=(30, 30))
        numFaces = len(face)
        self.ui.numberOfFaces_label.setText(str(numFaces))
        for (x, y, w, h) in face:
            exactFace = cv2.cvtColor(imageBGR[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
            cv2.rectangle(imageBGR , (x, y), (x + w, y + h), (0, 255, 0), 4)
        faceDetectedImg = cv2.cvtColor(imageBGR , cv2.COLOR_BGR2RGB)
        self.showDetectedFaces(faceDetectedImg)
       
       

    def showDetectedFaces(self, image):
        self.ui.graphicsLayout_AfterFaceDetection.clear()
        view_box = self.ui.graphicsLayout_AfterFaceDetection.addViewBox()
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        # Display the new image
        img_item = pg.ImageItem(image)
        view_box.addItem(img_item)


    