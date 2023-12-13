#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[3]:


# Load the dataframe with the predictions
df_val_combined = pd.read_csv('/projects/0/einf6214/output/2023-11-06_11-35-32_exponential_h-score/df_val_predictions.csv')

predicted_values = df_val_combined['prediction'].tolist()
predicted_values = [float(round(np.log2(x))) for x in predicted_values]

true_labels = df_val_combined['label'].tolist()
true_labels = [float(round(np.log2(x))) for x in true_labels]

# Define the class labels and the number of classes
class_labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
num_classes = len(class_labels)

# Create a confusion matrix to count occurrences of predicted vs. true labels
confusion_matrix = np.zeros((num_classes, num_classes))
for pred, true in zip(predicted_values, true_labels):
    pred_class = class_labels.index(pred)
    true_class = class_labels.index(true)
    confusion_matrix[true_class, pred_class] += 1

# Create the heatmap
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Greens)
plt.title("Confusion Matrix Heatmap")
plt.colorbar()

# Set the axis labels
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Set the axis ticks and labels
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# Display the values within the heatmap cells (optional)
for i in range(num_classes):
    for j in range(num_classes):
        value = int(confusion_matrix[i, j])
        plt.text(j, i, f"{value}", ha="center", va="center", color="black" if value < confusion_matrix.max() / 2 else "white")

plt.show()


# In[ ]:




