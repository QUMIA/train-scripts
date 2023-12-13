#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Load the dataframe with the predictions
df_val_combined = pd.read_csv('/projects/0/einf6214/output/2023-12-13_12-32-03_baseline-3/output/df_val_predictions.csv')

predicted_values = df_val_combined['prediction'].tolist()
predicted_values = [float(round(np.log(max(x, 1)) / np.log(4))) for x in predicted_values]

true_labels = df_val_combined['label'].tolist()
true_labels = [float(round(np.log(max(x, 1)) / np.log(4))) for x in true_labels]

# Define the class labels and the number of classes
class_labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
num_classes = len(class_labels)

# Create a confusion matrix to count occurrences of predicted vs. true labels
confusion_matrix = np.zeros((num_classes, num_classes))
for pred, true in zip(predicted_values, true_labels):
    pred_class = class_labels.index(pred)
    true_class = class_labels.index(true)
    confusion_matrix[true_class, pred_class] += 1

# Create a figure and axis
fig, ax = plt.subplots()

# Create the heatmap on the axis
cax = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Greens)
ax.set_title("Confusion Matrix Heatmap")
fig.colorbar(cax)

# Set the axis labels
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")

# Set the axis ticks and labels
tick_marks = np.arange(num_classes)
ax.set_xticks(tick_marks)
ax.set_xticklabels(class_labels)
ax.set_yticks(tick_marks)
ax.set_yticklabels(class_labels)
ax.invert_yaxis()

# Display the values within the heatmap cells (optional)
for i in range(num_classes):
    for j in range(num_classes):
        value = int(confusion_matrix[i, j])
        ax.text(j, i, f"{value}", ha="center", va="center", color="black" if value < confusion_matrix.max() / 2 else "white")

plt.show()

fig.savefig('my_heatmap.png')




