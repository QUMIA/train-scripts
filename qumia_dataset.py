"""
Dataset class that specifies how Pytorch should load and iterate the data.
Also keeps a reference to and applies the transformations to the image.
"""

import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import numpy as np

class QUMIA_Dataset(Dataset):
    def __init__(self, dataframe, transform=None, data_dir=None, num_classes=4, fuse_features=[]):
        self.data = dataframe
        self.transform = transform
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.fuse_features = fuse_features

        # Compute mean and std of fuse_features
        self.fuse_mean, self.fuse_std = self.compute_feature_stats()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Label
        label = QUMIA_Dataset.hscore_to_value(row["h_score"])
        if np.isnan(label):
            print(f"Label is NaN for {img_path}")

        # Image
        img_path = os.path.join(self.data_dir, row["exam_id"], row["image_file"])
        image = Image.open(img_path)
        image = np.array(image)
        if np.isnan(image).any():
            print(f"Image at {img_path} contains NaNs")
        if self.transform:
            image = self.transform(image=image)["image"]

        # Fuse features
        fuse_features = torch.tensor([row[feature_name] for feature_name in self.fuse_features], dtype=torch.float)
        if torch.isnan(fuse_features).any():
            print(f"Fuse features are NaN for {img_path}")
        # Normalize
        fuse_features = (fuse_features - self.fuse_mean) / self.fuse_std

        return {
            "image": image,
            "label": label,
            "fuse_features": fuse_features
        }

    def compute_feature_stats(self):
        features = self.data[self.fuse_features].values
        mean = torch.tensor(np.nanmean(features, axis=0), dtype=torch.float)
        std = torch.tensor(np.nanstd(features, axis=0), dtype=torch.float)
        return mean, std

    @staticmethod
    def value_to_hscore(y):
        return y
        # values = [0, 8, 12, 14]

        # # Handle cases where y is outside the bounds of the values list
        # if y <= values[0]:
        #     return 1.0
        # if y >= values[-1]:
        #     return 1.0 * len(values)

        # # Find the two closest numbers that y falls between
        # for i in range(len(values) - 1):
        #     if values[i] <= y <= values[i + 1]:
        #         lower_bound = values[i]
        #         upper_bound = values[i + 1]
        #         break

        # # Calculate the fractional position of y between these two numbers
        # fraction = (y - lower_bound) / (upper_bound - lower_bound)

        # # Return the interpolated index
        # return 1.0 * i + fraction + 1.0

    @staticmethod
    def hscore_to_value(hscore):
        return hscore
        # values = [0, 8, 12, 14]
        # index = int(hscore) - 1

        # # Handle cases where hscore is outside the valid index range
        # if index < 0:
        #     return values[0]
        # if index >= len(values):
        #     return values[-1]

        # return values[index]
