import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np

class QUMIA_Dataset(Dataset):
    def __init__(self, dataframe, transform=None, data_dir=None, num_classes=4):
        self.data = dataframe
        self.transform = transform
        self.data_dir = data_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.data_dir, row["exam_id"], row["image_file"])
        image = Image.open(img_path)
        label = QUMIA_Dataset.hscore_to_value(row["h_score"])

        image = np.array(image) 
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label

    @staticmethod
    def value_to_hscore(y):
        values = [0, 8, 12, 14]

        # Handle cases where y is outside the bounds of the values list
        if y <= values[0]:
            return 1
        if y >= values[-1]:
            return len(values)

        # Find the two closest numbers that y falls between
        for i in range(len(values) - 1):
            if values[i] <= y <= values[i + 1]:
                lower_bound = values[i]
                upper_bound = values[i + 1]
                break

        # Calculate the fractional position of y between these two numbers
        fraction = (y - lower_bound) / (upper_bound - lower_bound)

        # Return the interpolated index
        return i + fraction + 1

    @staticmethod
    def hscore_to_value(hscore):
        return [0, 8, 12, 14][int(hscore) - 1]
