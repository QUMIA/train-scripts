#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
print("Python version: " + sys.version)


# In[2]:


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


# In[3]:


# Data directories
data_dir = '/projects/0/einf6214/data'
data_dir_images = os.path.join(data_dir, 'merged')

# Ouput directory
output_dir = '/projects/0/einf6214/output'
if not 'run_dir' in locals():
    run_dir = os.path.join(output_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(run_dir)


# In[5]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


# In[6]:


# Read data
df_train = pd.read_csv(os.path.join(data_dir, 'split_train.csv'))
df_val = pd.read_csv(os.path.join(data_dir, 'split_val.csv'))
print(df_train.shape, df_val.shape)


# In[7]:


# Transform applied to each image
data_transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.225]),
])


# In[8]:


# Create dataset and dataloader for the train data
train_dataset = QUMIA_Dataset(df_train, transform=data_transform, data_dir=data_dir_images)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)

# Create dataset and dataloader for the validation data (no shuffle)
validation_dataset = QUMIA_Dataset(df_val, transform=data_transform, data_dir=data_dir_images)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=1)


# In[13]:


# Instantiate and prepare model
model = QUMIA_Model()
model.to(device)

# Print a summary of the model
summary(model, (1, 224, 224), device=device.type)

# Loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[40]:


def train():
    num_epochs = 10  # Number of training epochs

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

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
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(run_dir, f'model_epoch_{epoch}.pth'))

        make_predictions(model, validation_loader)

    # Save the model and weights
    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))


# In[34]:


def make_predictions(model, dataloader, n_batches=None):
    model.eval()  # Set the model to evaluation mode

    predictions = []
    labels = []
    
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

        print(f"Validation loss: {running_loss / len(validation_loader):.4f}")
    
    return torch.cat(predictions), torch.cat(labels)


# In[35]:


def validate(n_batches=None):
    # Load the model and weights
    model = QUMIA_Model()
    model.to(device)
    model.load_state_dict(torch.load('final_model_fixed_split.pth', map_location=device))

    # Make predictions on the validation set
    predictions, labels = make_predictions(model, validation_loader, n_batches)

    # Convert predictions and labels to numpy arrays
    predictions = predictions.numpy().flatten()
    rounded_predictions = np.round(predictions)
    labels = labels.numpy().flatten()
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
    df_val_combined.to_csv(os.path.join(run_dir, 'df_val_predictions.csv'), index=False)

    return df_val_combined


# In[36]:


#df_val_combined['label'].equals(df_val_combined['h_score'].astype('float32'))


# In[38]:


# Main function that checks the argument passed to the script and calls the appropriate function
def main():
    if len(sys.argv) != 2:
        print("Usage: python train.py {train|validate}")
        exit(1)

    command = sys.argv[1]
    if (command == "train"):
        train()
    elif (command == "validate"):
        validate()

if __name__ == '__main__' and '__file__' in globals():
    main()
