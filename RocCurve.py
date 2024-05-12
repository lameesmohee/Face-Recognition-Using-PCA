# # from sklearn.metrics import roc_curve, auc
# # import matplotlib.pyplot as plt

# # class ROCPlotter:
# #     def __init__(self, main_tab_widget):
# #         self.main_tab_widget = main_tab_widget
# #         self.UI=self.main_tab_widget
        
# #     def plot_ROC(self, actual_labels, predicted_labels):
# #         fpr, tpr, thresholds = roc_curve(actual_labels, predicted_labels)
# #         roc_auc = auc(fpr, tpr)

# #         # Clear the previous plot
# #         self.UI.graphicsLayout_BeforeFaceRecognition_2.clear()

# #         # Create a plot item
# #         plot_item = self.UI.graphicsLayout_BeforeFaceRecognition_2.addPlot(title="ROC Curve")
# #         plot_item.plot(fpr, tpr, pen='b', name='ROC curve (area = %0.2f)' % roc_auc)
# #         plot_item.plot([0, 1], [0, 1], pen='r', linestyle='--')

# #         # Set axis labels and title
# #         plot_item.setLabel('left', 'True Positive Rate')
# #         plot_item.setLabel('bottom', 'False Positive Rate')
# #         plot_item.setTitle('Receiver Operating Characteristic (ROC) Curve')



# import numpy as np
# import pyqtgraph as pg
# from sklearn.metrics import roc_curve


# class ROCPlotter:
#     def __init__(self, main_tab_widget):
#         self.main_tab_widget = main_tab_widget
#         self.UI=self.main_tab_widget

#     def plot_ROC(self, true_labels, predicted_scores):
#         # Convert true labels to binary (1 for positive, 0 for negative)
#         true_labels_binary = [1 if label else 0 for label in true_labels]

#         # Calculate FPR and TPR
#         fpr, tpr, _ = roc_curve(true_labels_binary, predicted_scores)

#         # Plot ROC curve
#         self.UI.graphicsLayout_BeforeFaceRecognition_2.clear()
#         roc_plot = self.UI.graphicsLayout_BeforeFaceRecognition_2.addPlot(title="ROC Curve")
#         roc_plot.plot(fpr, tpr, pen=pg.mkPen('b', width=2))
#         roc_plot.plot([0, 1], [0, 1], pen=pg.mkPen('r', width=2), style=pg.QtCore.Qt.DotLine)
#         roc_plot.setLabel('left', "True Positive Rate")
#         roc_plot.setLabel('bottom', "False Positive Rate")
#         roc_plot.showGrid(True, True)


# def calculate_ROC(self):
#         # Initialize lists to store TPR and FPR
#         tpr_list = []
#         fpr_list = []

#         # Get the total number of positive and negative samples
#         total_positives = sum(self.true_labels)
#         total_negatives = len(self.true_labels) - total_positives

#         # Sort the predicted scores and true labels accordingly
#         sorted_indices = np.argsort(self.predicted_scores)
#         sorted_labels = np.array(self.true_labels)[sorted_indices]
#         sorted_scores = np.array(self.predicted_scores)[sorted_indices]

#         # Initialize variables to store the previous FPR and TPR
#         prev_fpr = 0
#         prev_tpr = 0

#         # Calculate TPR and FPR at different threshold levels
#         for threshold in np.unique(sorted_scores):
#             # Thresholding the predicted scores
#             thresholded_predictions = (sorted_scores >= threshold).astype(int)
            
#             # Count true positives, false positives, true negatives, and false negatives
#             tp = np.sum((thresholded_predictions == 1) & (sorted_labels == 1))
#             fp = np.sum((thresholded_predictions == 1) & (sorted_labels == 0))
#             tn = total_negatives - fp
#             fn = total_positives - tp

#             # Calculate TPR and FPR
#             tpr = tp / (tp + fn)
#             fpr = fp / (fp + tn)

#             # Append TPR and FPR to the lists
#             tpr_list.append(tpr)
#             fpr_list.append(fpr)

#             # Interpolate to ensure smooth curve
#             if fpr != prev_fpr:
#                 tpr_list.append(prev_tpr + (tpr - prev_tpr) * (prev_fpr - fpr) / (prev_fpr - fpr))
#                 fpr_list.append(prev_fpr)
            
#             # Update previous valuesa
#             prev_fpr = fpr
#             prev_tpr = tpr

#         # Add the last point
#         tpr_list.append(1)
#         fpr_list.append(1)

#         return tpr_list, fpr_list
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Simulated dataset: pairs of face images and labels
# In a real scenario, you would load actual image data and compute similarity scores
n_samples = 1000
np.random.seed(42)
# Simulated labels: 1 if same person, 0 if different people
labels = np.random.choice([0, 1], size=n_samples)

# Simulated similarity scores: higher scores indicate higher similarity
# In practice, these would be computed using a face recognition model
similarity_scores = np.random.rand(n_samples)

# Split the dataset into train and test sets
labels_train, labels_test, scores_train, scores_test = train_test_split(labels, similarity_scores, test_size=0.3, random_state=42)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(labels_test, scores_test)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) for Face Recognition')
plt.legend(loc="lower right")
plt.show()