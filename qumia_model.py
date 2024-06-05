import torch.nn as nn
import torch

class QUMIA_Model(nn.Module):
    def __init__(self, in_channels=1, image_size=448, n_layers=5, first_out_channels=32, fully_connected_size=256, fuse_features_size=2):
        super().__init__()

        # Convolutional layers with batch normalization
        self.conv_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        out_channels = first_out_channels
        for i in range(n_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            self.batch_norm_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            out_channels *= 2

        # Pooling layer (used after each convolutional layer)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer (fc1) and combining everything (fc2)
        reduced_size = image_size // (2 ** n_layers)
        self.conv_out_size = (out_channels // 2) * reduced_size * reduced_size
        self.fc1_in_size = self.conv_out_size + fuse_features_size
        self.fc1 = nn.Linear(self.fc1_in_size, fully_connected_size)
        self.bn_fc1 = nn.BatchNorm1d(fully_connected_size)
        self.fc2 = nn.Linear(fully_connected_size, 1)

    def forward(self, x, fuse_features):
        # Convolutional layers with batch normalization
        for conv, bn in zip(self.conv_layers, self.batch_norm_layers):
            x = self.pool(nn.functional.relu(bn(conv(x))))
        
        # Flatten output of convolutional layers
        x = x.view(-1, self.conv_out_size)

        # Add the fuse features to the fully connected layer
        x = torch.cat((x, fuse_features), dim=1)

        # Fully connected layers
        x = nn.functional.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)
        return x
