import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )

    def forward(self, x):
        return self.conv_block(x)


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.dense_block = nn.Sequential(
            nn.Linear(in_features, out_features),
            # nn.BatchNorm1d(out_features),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.dense_block(x)


class Model(nn.Module):
    name = "basic"
    is_replaced = True
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(12, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)

        self.classifier1 = DenseBlock(1024+7, 512)
        self.classifier2 = DenseBlock(512, 512)
        self.classifier3 = nn.Linear(512, 101)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, spec_tensor):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x) 
        x = self.conv4(x)
        x = x.view(x.size(0), -1)

        # http://kaga100man.com/2019/03/25/post-102/
        x = torch.cat([x, spec_tensor], dim=1)


        x = self.classifier1(x)
        x = self.classifier2(x)
        x = self.classifier3(x)
        x = self.relu(x)

        return x