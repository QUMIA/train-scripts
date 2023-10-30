#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
print("Python version: " + sys.version)


# In[4]:


import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler
from torchsummary import summary
from qumia_dataset import QUMIA_Dataset
from qumia_model import QUMIA_Model
from tqdm import tqdm

data_dir = '/projects/0/einf6214/data'
data_dir_images = os.path.join(data_dir, 'merged')


# In[6]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: " + str(device))


# In[7]:


# Read data
df_train = pd.read_csv(os.path.join(data_dir, 'split_train.csv'))
df_val = pd.read_csv(os.path.join(data_dir, 'split_val.csv'))
print(df_train.shape, df_val.shape)


# In[8]:


# Transform applied to each image
data_transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.225]),
])


# In[9]:


# Create dataset and dataloader for the train data
train_dataset = QUMIA_Dataset(df_train, transform=data_transform, data_dir=data_dir_images)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1)

# Create dataset and dataloader for the validation data (no shuffle)
validation_dataset = QUMIA_Dataset(df_val, transform=data_transform, data_dir=data_dir_images)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, num_workers=1)


# In[10]:


# Instantiate model
model = QUMIA_Model()

# Set mode to train (vs. eval for inference)
model.train()

# Move model to device (GPU or CPU)
model.to(device)

# Print a summary of the model
summary(model, (1, 224, 224))


# In[11]:


import torch.optim as optim

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[12]:


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
        torch.save(model.state_dict(), os.path.join('tmp', f'model_epoch_{epoch}.pth'))

        make_predictions(model, validation_loader)

    # Save the model and weights
    torch.save(model.state_dict(), 'final_model.pth')


# In[15]:


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


# In[58]:


#df_val_combined['label'].equals(df_val_combined['h_score'].astype('float32'))


# In[ ]:


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

