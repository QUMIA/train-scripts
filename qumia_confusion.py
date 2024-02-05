import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
import csv


def array_to_csv_with_headers(array, path, row_headers, col_headers):
    """
    Converts a 2D NumPy array to a CSV file with row and column headers.

    :param array: 2D NumPy array to be converted.
    :param file_name: Name of the CSV file to save the array to.
    :param row_headers: List of row headers.
    :param col_headers: List of column headers.
    """
    # Ensure the array is 2D
    if len(array.shape) != 2:
        raise ValueError("Only 2D arrays are supported")

    # Check if the headers match the array dimensions
    if len(row_headers) != array.shape[0] or len(col_headers) != array.shape[1]:
        raise ValueError("Headers do not match the dimensions of the array")

    # Write the array to a CSV file
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)

        # Write column headers
        writer.writerow([''] + col_headers)

        # Write rows with row headers
        for header, row in zip(row_headers, array):
            writer.writerow([header] + list(row))


def create_confusion_matrix(predicted_values, true_labels, set_type, output_dir):
    # Define the class labels and the number of classes
    class_labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    num_classes = len(class_labels)

    # Create a confusion matrix to count occurrences of predicted vs. true labels
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
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

    # Save the figure to a file
    image_path = os.path.join(output_dir, f'confusion_matrix_{set_type}.png')
    fig.savefig(image_path)

    # Log the confusion matrix as an image artifact to W&B
    wandb.log({f"confusion_matrix_{set_type}": wandb.Image(image_path)})

    # Save the confusion matrix to a csv file
    row_headers = [f"True_{label}" for label in class_labels]
    col_headers = [f"Pred_{label}" for label in class_labels]
    csv_path = os.path.join(output_dir, f'confusion_matrix_{set_type}.csv')
    array_to_csv_with_headers(confusion_matrix, csv_path, row_headers, col_headers)
