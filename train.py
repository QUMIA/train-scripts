#!/usr/bin/env python
# coding: utf-8

import sys
print("Python version: " + sys.version)


import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torchsummary import summary
from torchvision import transforms
from qumia_dataset import QUMIA_Dataset
from qumia_model import QUMIA_Model
from tqdm import tqdm
from datetime import datetime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import copy
import matplotlib.pyplot as plt
import wandb
from dotenv import load_dotenv


# Load environment variables from .env file
if load_dotenv():
    sessionLabel = os.getenv('SESSION_LABEL')
else:
    sessionLabel = None
print(sessionLabel)


# Data directories
data_dir = '/projects/0/einf6214/data'
data_dir_images = os.path.join(data_dir, 'merged')

# Output dir
output_dir = 'output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="qumia",

    name=sessionLabel,
    
    # track hyperparameters and run metadata
    config={
        "learning_rate": 0.001,
        "architecture": "CNN",
        "epochs": 20,
    }
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


# Read data
df_train = pd.read_csv(os.path.join(data_dir, 'split_train.csv'))
df_val = pd.read_csv(os.path.join(data_dir, 'split_val.csv'))
print(df_train.shape, df_val.shape)


elastic_alpha = 480.0

train_transform = A.Compose(
    [
        #A.HorizontalFlip(p=0.5),
        #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.7),
        A.Resize(448, 448),
        #A.ElasticTransform(p=1, alpha=elastic_alpha, sigma=elastic_alpha * 0.07, alpha_affine=elastic_alpha * 0.05),
        #A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.5,), std=(0.225,)),
        ToTensorV2(),
    ]
)

validation_transform = A.Compose(
    [
        A.Resize(448, 448),
        A.Normalize(mean=(0.5,), std=(0.225,)),
        ToTensorV2(),
    ]
)


# Create dataset and dataloader for the train data
train_dataset = QUMIA_Dataset(df_train, transform=train_transform, data_dir=data_dir_images)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)

# Create dataset and dataloader for the validation data (no shuffle)
validation_dataset = QUMIA_Dataset(df_val, transform=validation_transform, data_dir=data_dir_images)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=8)


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

visualize_augmentations(train_dataset)


# Instantiate and prepare model
model = QUMIA_Model()
model.to(device)

# Print a summary of the model
summary(model, (1, 448, 448), device=device.type)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    num_epochs = 20  # Number of training epochs

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0

        n_seconds = 10  # Print every n seconds
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            
            # Reshape labels to match output of model
            labels = labels.view(-1, 1).float()

            # Move input and label tensors to the default device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # print the shape of the input and label tensors
            #print(inputs.shape, labels.shape)
            #print(inputs.dtype, labels.dtype)

            optimizer.zero_grad()

            outputs = model(inputs)
            # print(outputs.shape)
            # print(outputs.dtype)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:  # Print every 100 mini-batches
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch}.pth'))

        _, _, train_loss = make_predictions(model, train_loader)
        print(f"Train loss: {train_loss:.4f}")
        _, _, validation_loss = make_predictions(model, validation_loader)
        print(f"Validation loss: {validation_loss:.4f}")

        wandb.log({"train-loss": train_loss, "validation-loss": validation_loss, "epoch": epoch})

    # Save the model and weights
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))

    # Do the final validation run (saving the predictions)
    validate(model)

    wandb.finish()


def make_predictions(model, dataloader, n_batches=None):
    model.eval()  # Set the model to evaluation mode

    predictions = []
    labels = []
    loss = None

    with torch.no_grad():
        running_loss = 0.0
        for index, batch in enumerate(dataloader, 0): # tqdm(dataloader, total=len(dataloader), desc="Performing predictions on validation data"):
            inputs, batch_labels = batch
            batch_labels = batch_labels.view(-1, 1).float()

            # Move input and label tensors to the default device
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Save predictions and labels
            predictions.append(outputs)
            labels.append(batch_labels)

            # Compute loss
            loss = criterion(outputs, batch_labels)
            running_loss += loss.item()

            index += 1
            if n_batches is not None and index > n_batches:
                break

        loss = running_loss / len(dataloader)

    return torch.cat(predictions), torch.cat(labels), loss


def validate(model, n_batches=None):
    # Make predictions on the validation set
    predictions, labels, validation_loss = make_predictions(model, validation_loader, n_batches)
    print(f"Validation loss: {validation_loss:.4f}")
    
    # Convert predictions and labels to numpy arrays
    predictions = predictions.cpu().numpy().flatten()
    rounded_predictions = np.round(predictions)
    labels = labels.cpu().numpy().flatten()
    print(predictions.shape, labels.shape)

    # We might only have predictions for a number of batches, so we need to trim the dataframe
    df_val_combined = df_val.iloc[:predictions.shape[0]].copy()

    # Combine the original dataframe with the predictions
    df_val_combined['prediction'] = predictions
    df_val_combined['rounded_prediction'] = rounded_predictions
    df_val_combined['label'] = labels # redundant, but we could detect a mismatch with the inputs maybe

    # As a sanity check, see if the labels match the original input rows
    match = df_val_combined['label'].equals(df_val_combined['h_score'].astype('float32'))
    print(f"Labels match: {match}")
    if not match:
        print("Possible mismatch between labels and inputs!")
        #raise Exception("Mismatch between labels and inputs")

    # Save the dataframe to a csv file
    df_val_combined.to_csv(os.path.join(output_dir, 'df_val_predictions.csv'), index=False)

    return df_val_combined


#df_val_combined['label'].equals(df_val_combined['h_score'].astype('float32'))


def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py {train|validate}")
        exit(1)

    command = sys.argv[1]
    if (command == "train"):
        train()
    elif (command == "validate"):
        validate() #TODO: pass model as argument

# Check if we are running as a script and not in a notebook
if __name__ == '__main__' and '__file__' in globals():
    main()
