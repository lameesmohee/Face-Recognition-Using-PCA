# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt

# class ROCPlotter:
#     def __init__(self, main_tab_widget):
#         self.main_tab_widget = main_tab_widget
#         self.UI=self.main_tab_widget
        
#     def plot_ROC(self, actual_labels, predicted_labels):
#         fpr, tpr, thresholds = roc_curve(actual_labels, predicted_labels)
#         roc_auc = auc(fpr, tpr)

#         # Clear the previous plot
#         self.UI.graphicsLayout_BeforeFaceRecognition_2.clear()

#         # Create a plot item
#         plot_item = self.UI.graphicsLayout_BeforeFaceRecognition_2.addPlot(title="ROC Curve")
#         plot_item.plot(fpr, tpr, pen='b', name='ROC curve (area = %0.2f)' % roc_auc)
#         plot_item.plot([0, 1], [0, 1], pen='r', linestyle='--')

#         # Set axis labels and title
#         plot_item.setLabel('left', 'True Positive Rate')
#         plot_item.setLabel('bottom', 'False Positive Rate')
#         plot_item.setTitle('Receiver Operating Characteristic (ROC) Curve')



import numpy as np
import pyqtgraph as pg
from sklearn.metrics import roc_curve


class ROCPlotter:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.UI=self.main_tab_widget

    def plot_ROC(self, true_labels, predicted_scores):
        # Convert true labels to binary (1 for positive, 0 for negative)
        true_labels_binary = [1 if label else 0 for label in true_labels]

        # Calculate FPR and TPR
        fpr, tpr, _ = roc_curve(true_labels_binary, predicted_scores)

        # Plot ROC curve
        self.UI.graphicsLayout_BeforeFaceRecognition_2.clear()
        roc_plot = self.UI.graphicsLayout_BeforeFaceRecognition_2.addPlot(title="ROC Curve")
        roc_plot.plot(fpr, tpr, pen=pg.mkPen('b', width=2))
        roc_plot.plot([0, 1], [0, 1], pen=pg.mkPen('r', width=2), style=pg.QtCore.Qt.DotLine)
        roc_plot.setLabel('left', "True Positive Rate")
        roc_plot.setLabel('bottom', "False Positive Rate")
        roc_plot.showGrid(True, True)


