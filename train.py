#!/usr/bin/env python
# coding: utf-8

import sys
print("Python version: " + sys.version)


import os
import pandas as pd
import copy
import matplotlib.pyplot as plt
from dotenv import load_dotenv

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from qumia_dataset import QUMIA_Dataset
from qumia_model import QUMIA_Model
from qumia_core import QUMIA_Trainer, train, validate


# Load environment variables from .env file
if load_dotenv():
    sessionLabel = os.getenv('SESSION_LABEL')
else:
    sessionLabel = None
print(sessionLabel)


def is_running_as_script():
    return __name__ == '__main__' and '__file__' in globals()


image_size = 448
image_channels = 1
config={
    "learning_rate": 0.001,
    "model": "QUMIA_Model",
    "epochs": 20,
    "image_size": image_size,
    "image_channels": image_channels,
    "model_layers": 5,
    "model_first_out_channels": 32,
    "model_fully_connected_size": 256,
}


# start a new wandb run to track this script
wandb.init(
    project="qumia",
    name=sessionLabel,
    config=config,
)


# Data directories
data_dir = '/projects/0/einf6214/data'
data_dir_images = os.path.join(data_dir, 'masked')

# Output dir (relative to code; we assume a dedicated directory with the copied code for each session, see run_session.sh)
output_dir = 'output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


# Read data
df_train = pd.read_csv(os.path.join(data_dir, 'split_train.csv'))
df_val = pd.read_csv(os.path.join(data_dir, 'split_val.csv'))
df_test = pd.read_csv(os.path.join(data_dir, 'split_test.csv'))
print(df_train.shape, df_val.shape, df_test.shape)


elastic_alpha = 480.0

train_transform = A.Compose(
    [
        #A.HorizontalFlip(p=0.5),
        #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.7),
        A.Resize(image_size, image_size),
        #A.ElasticTransform(p=1, alpha=elastic_alpha, sigma=elastic_alpha * 0.07, alpha_affine=elastic_alpha * 0.05),
        #A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.5,), std=(0.225,)),
        ToTensorV2(),
    ]
)

evaluation_transform = A.Compose(
    [
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.5,), std=(0.225,)),
        ToTensorV2(),
    ]
)


fuse_features = ["bmi", "Age_exam"]
#fuse_features = []
use_subset = not is_running_as_script()

# Create dataset and dataloader for the train data
train_dataset = QUMIA_Dataset(df_train, transform=train_transform, data_dir=data_dir_images, fuse_features=fuse_features)
train_subset = Subset(train_dataset, range(100))
train_loader = DataLoader(train_subset if use_subset else train_dataset, batch_size=32, shuffle=True, num_workers=8)

# Create dataset and dataloader for the validation data (no shuffle)
validation_dataset = QUMIA_Dataset(df_val, transform=evaluation_transform, data_dir=data_dir_images, fuse_features=fuse_features)
validation_subset = Subset(validation_dataset, range(30))
validation_loader = DataLoader(validation_subset if use_subset else validation_dataset, batch_size=32, shuffle=False, num_workers=8)

# Create dataset and dataloader for the test data (no shuffle)
test_dataset = QUMIA_Dataset(df_test, transform=evaluation_transform, data_dir=data_dir_images, fuse_features=fuse_features)
test_subset = Subset(test_dataset, range(30))
test_loader = DataLoader(test_subset if use_subset else test_dataset, batch_size=32, shuffle=False, num_workers=8)


def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx+100]
        ax.ravel()[i].imshow(image, cmap='gray')
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()

#visualize_augmentations(train_dataset)


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

#print(value_to_hscore(13))


def create_model():
    model = QUMIA_Model(config["image_channels"], image_size, config["model_layers"], 
                        config["model_first_out_channels"], config["model_fully_connected_size"],
                        len(fuse_features))
    model.to(device)
    return model


# Instantiate and prepare model
model = create_model()

# Print a summary of the model
# with feature fusion:
# summary(model, input_data=[(1, image_channels, image_size, image_size), (1, 2)], device=device.type)

#summary(model, (image_channels, image_size, image_size), device=device.type)


# Loss function
#criterion = torch.nn.MSELoss()

# class_weights = torch.tensor([0.01195304017, 0.03527882809, 0.08486611199, 0.8679020198])
# class_weights = class_weights.to(device)

# def weighted_mse_loss(input, target):
#     assert input.shape == target.shape, "Input and target must have the same shape"

#     # Assign weights based on the target class
#     # This assumes targets are 1.0, 2.0, 3.0, and 4.0 for the classes
#     sample_weights = class_weights[target.long() - 1]

#     # Calculate MSE loss for each sample
#     mse = torch.nn.functional.mse_loss(input, target, reduction='none')

#     # Weight the MSE loss by the sample weights
#     weighted_mse = mse * sample_weights

#     # Return the mean loss
#     return weighted_mse.mean()

criterion = torch.nn.MSELoss()


# Optimizer
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])


# Create a trainer that holds all the objects
trainer = QUMIA_Trainer(df_train, df_val, df_test, train_loader, validation_loader, test_loader,
                        device, model, criterion, optimizer, output_dir)


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py {train|validate}")
        exit(1)

    command = sys.argv[1]
    if (command == "train"):
        train(config["epochs"], trainer)
    elif (command == "validate"):
        validate(trainer, set_type="validation")

# Check if we are running as a script and not in a notebook
if is_running_as_script():
    main()

