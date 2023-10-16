#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from qumia_dataset import QUMIA_Dataset
from FullyConvolutionalResnet18 import FullyConvolutionalResnet18

data_dir = '/projects/0/einf6214/data'
data_dir_images = os.path.join(data_dir, 'merged')


# In[2]:


# auto reload modules
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


# Read data
df_train = pd.read_csv(os.path.join(data_dir, 'split_train.csv'))
df_val = pd.read_csv(os.path.join(data_dir, 'split_val.csv'))
print(df_train.shape, df_val.shape)


# In[4]:


data_transform = transforms.Compose([
    #transforms.Grayscale(num_output_channels=3),  # Convert to RGB
    #transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.225]),
])


# In[5]:


train_dataset = QUMIA_Dataset(df_train, transform=data_transform, data_dir=data_dir_images)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

validation_dataset = QUMIA_Dataset(df_val, transform=data_transform, data_dir=data_dir_images)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, num_workers=1)


# In[6]:


model = FullyConvolutionalResnet18(pretrained=False).eval()
summary(model, (1, 400, 400))


# In[ ]:


import torch.optim as optim

criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[ ]:


num_epochs = 10  # Number of training epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # print the shape of the input and label tensors
        print(inputs.shape, labels.shape)
        print(inputs.dtype, labels.dtype)

        optimizer.zero_grad()

        outputs = model(inputs)
        print(outputs.shape)
        print(outputs.dtype)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i + 1) % 100 == 0:  # Print every 100 mini-batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.3f}")
            running_loss = 0.0


# In[ ]:




