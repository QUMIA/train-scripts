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
        label = 2 ** row["h_score"]  # h-score starts at 1, index at 0
        # one_hot_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes) \
        #     .to(torch.float32)

        image = np.array(image) 
        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label
