from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        n_classes = 10
        kernel_size = 3
        in_channels = 1
        conv1_out_channels = 4
        conv2_out_channels = 8
        conv3_out_channels = 16
        self.conv1 = nn.Conv2d(in_channels, conv1_out_channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(conv1_out_channels)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size)
        self.bn2 = nn.BatchNorm2d(conv2_out_channels)
        self.conv3 = nn.Conv2d(conv2_out_channels, conv3_out_channels, kernel_size)
        self.fc = nn.Linear(16, n_classes)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Input should have dimensions (batch_size, height, width)")
        x = x.view(x.shape[0], 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
