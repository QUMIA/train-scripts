import torch.nn as nn
import torch.nn.functional as F

class MaskedClassifier(nn.Module):

    def __init__(self, vector_length=16):
        super(MaskedClassifier, self).__init__()

        # Some variables
        conv_out_channels = 64
        self.flat_features = conv_out_channels * vector_length

        # Convolution layer with 1x1 kernel
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv_out_channels, kernel_size=1, stride=1, padding=0)

        # Fully connected layer from the flattened convolution output (16*16) to 100 nodes
        self.fc1 = nn.Linear(self.flat_features, 100)

        # Output layer
        self.fc2 = nn.Linear(100, 1)


    def forward(self, x):
        # Apply 1x1 convolution
        x = F.relu(self.conv1(x))

        # Flatten the output for the fully connected layer
        x = x.view(-1, self.flat_features)

        # Apply the first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))

        # Apply the second fully connected layer for classification
        x = self.fc2(x)
        return x
