import torch.nn as nn

class QUMIA_Model(nn.Module):

    def __init__(self, in_channels=1, image_size=448, n_layers=5, first_out_channels=32, fully_connected_size=256): 
        super().__init__()

        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        out_channels = first_out_channels
        for i in range(n_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            in_channels = out_channels
            out_channels *= 2

        # Pooling layer (used after each convolutional layer)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer (fc1) and combining everything (fc2)
        reduced_size = image_size // (2 ** n_layers)
        self.fc1_in_size = (out_channels // 2) * reduced_size * reduced_size
        self.fc1 = nn.Linear(self.fc1_in_size, fully_connected_size)
        self.fc2 = nn.Linear(fully_connected_size, 1)

    def forward(self, x):
        # Convolutional layers
        for conv in self.conv_layers:
            x = self.pool(nn.functional.relu(conv(x)))

        # Flatten output of convolutional layers
        x = x.view(-1, self.conv_out_size)

        # Add the fuse features to the fully connected layer
        # x = torch.cat((x, fuse_features), dim=1)

        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
