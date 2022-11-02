import torch
import torch.nn as nn


class UpChannel(nn.Module):
    def __init__(self, in_channels, out_channels, is_donwsample=True):
        super().__init__()
        self.is_donwsample = is_donwsample

        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):

        x = self.conv(x)
        x = self.relu(x)

        if self.is_donwsample:
            x = self.downsample(x)

        return x


class PlaneBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv3d(channels, channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

        self.relu = nn.ReLU(inplace=True)

        # self.shortcut = self._shortcut(channels, channels)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.relu(x)

        return x

    # def _shortcut(self, in_channels, out_chammels):
    #     if in_channels != out_chammels:
    #         return nn.Conv3d(in_channels, out_chammels, kernel_size=1, padding=0)
    #     else:
    #         return lambda x: x


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


class ResNet3d(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(11, 64, kernel_size=7, stride=2, padding=3, bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )

        self.conv2_x = nn.Sequential(
            PlaneBlock(64),
            PlaneBlock(64),
            PlaneBlock(64),
        )

        self.upchannel1 = UpChannel(64, 128)

        self.conv3_x = nn.Sequential(
            PlaneBlock(128),
            PlaneBlock(128),
            PlaneBlock(128),
            PlaneBlock(128),
        )

        self.upchannel2 = UpChannel(128, 256)

        self.conv4_x = nn.Sequential(
            PlaneBlock(256),
            PlaneBlock(256),
            PlaneBlock(256),
            PlaneBlock(256),
            PlaneBlock(256),
            PlaneBlock(256),
        )

        self.upchannel3 = UpChannel(256, 512)

        self.conv5_x = nn.Sequential(
            PlaneBlock(512),
            PlaneBlock(512),
            PlaneBlock(512),
        )

        self.avg_pool = nn.AvgPool3d(kernel_size=(2, 2, 1))

        self.classifier = nn.Sequential(
            DenseBlock(512+7, 256),
            nn.Linear(256, 101),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, spec_tensor):
        x = self.conv1(x)    # torch.Size([2, 64, 20, 20, 15])
        x = self.conv2_x(x)  # torch.Size([2, 64, 20, 20, 15])
        x = self.upchannel1(x)
        x = self.conv3_x(x)  # torch.Size([2, 128, 10, 10, 7])
        x = self.upchannel2(x)
        x = self.conv4_x(x)  # torch.Size([2, 256, 5, 5, 3])
        x = self.upchannel3(x)
        x = self.conv5_x(x)  # torch.Size([2, 512, 2, 2, 1])

        x = self.avg_pool(x)    # torch.Size([2, 512, 1, 1, 1])
        x = x.reshape(x.size(0), -1)    # torch.Size([2, 512])

        # http://kaga100man.com/2019/03/25/post-102/
        x = torch.cat([x, spec_tensor], dim=1)

        x = self.classifier(x)

        return x
