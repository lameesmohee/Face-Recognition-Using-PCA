from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class ROCPlotter:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.UI=self.main_tab_widget
        
    def plot_ROC(self, actual_labels, predicted_labels):
        fpr, tpr, thresholds = roc_curve(actual_labels, predicted_labels)
        roc_auc = auc(fpr, tpr)

        # Clear the previous plot
        self.UI.graphicsLayout_BeforeFaceRecognition_2.clear()

        # Create a plot item
        plot_item = self.UI.graphicsLayout_BeforeFaceRecognition_2.addPlot(title="ROC Curve")
        plot_item.plot(fpr, tpr, pen='b', name='ROC curve (area = %0.2f)' % roc_auc)
        plot_item.plot([0, 1], [0, 1], pen='r', linestyle='--')

        # Set axis labels and title
        plot_item.setLabel('left', 'True Positive Rate')
        plot_item.setLabel('bottom', 'False Positive Rate')
        plot_item.setTitle('Receiver Operating Characteristic (ROC) Curve')
