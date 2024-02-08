"""
This file contains code that allows setting the transformation of the image.
The file contrains classes and functions that specify how Pytorch should load and iterate the data.
"""

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
    def value_to_hscore(value):
        return np.log(max(value, 1)) / np.log(4)

    @staticmethod
    def hscore_to_value(hscore):
        return np.power(4, hscore)
