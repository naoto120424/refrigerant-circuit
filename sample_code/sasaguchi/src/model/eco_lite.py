import torch
from torch import nn

class BasicConv(nn.Module):
    '''ECOの2D Netモジュールの最初のモジュール'''

    def __init__(self):
        super(BasicConv, self).__init__()

        self.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3)
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(192)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = self.relu(out)

        return out

class InceptionA(nn.Module):
    '''InceptionA'''

    def __init__(self):
        super(InceptionA, self).__init__()
        self.conv1_1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(96)
        self.conv3_3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(96)
        self.pool4_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_1_bn(out1)
        out1 = self.relu(out1)

        out2 = self.conv2_1(x)
        out2 = self.conv2_1_bn(out2)
        out2 = self.relu(out2)
        out2 = self.conv2_2(out2)
        out2 = self.conv2_2_bn(out2)
        out2 = self.relu(out2)

        out3 = self.conv3_1(x)
        out3 = self.conv3_1_bn(out3)
        out3 = self.relu(out3)
        out3 = self.conv3_2(out3)
        out3 = self.conv3_2_bn(out3)
        out3 = self.relu(out3)
        out3 = self.conv3_3(out3)
        out3 = self.conv3_3_bn(out3)
        out3 = self.relu(out3)

        out4 = self.pool4_1(x)
        out4 = self.conv4_2(out4)
        out4 = self.conv4_2_bn(out4)
        out4 = self.relu(out4)

        outputs = [out1, out2, out3, out4]

        return torch.cat(outputs, 1)

class InceptionB(nn.Module):
    '''InceptionB'''

    def __init__(self):
        super(InceptionB, self).__init__()
        self.conv1_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(96)
        self.conv3_1 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.conv3_1_bn = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(96)
        self.conv3_3 = nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        self.conv3_3_bn = nn.BatchNorm2d(96)
        self.pool4_1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out1 = self.conv1_1(x)
        out1 = self.conv1_1_bn(out1)
        out1 = self.relu(out1)

        out2 = self.conv2_1(x)
        out2 = self.conv2_1_bn(out2)
        out2 = self.relu(out2)
        out2 = self.conv2_2(out2)
        out2 = self.conv2_2_bn(out2)
        out2 = self.relu(out2)

        out3 = self.conv3_1(x)
        out3 = self.conv3_1_bn(out3)
        out3 = self.relu(out3)
        out3 = self.conv3_2(out3)
        out3 = self.conv3_2_bn(out3)
        out3 = self.relu(out3)
        out3 = self.conv3_3(out3)
        out3 = self.conv3_3_bn(out3)
        out3 = self.relu(out3)

        out4 = self.pool4_1(x)
        out4 = self.conv4_2(out4)
        out4 = self.conv4_2_bn(out4)
        out4 = self.relu(out4)

        outputs = [out1, out2, out3, out4]

        return torch.cat(outputs, 1)

class InceptionC(nn.Module):
    def __init__(self):
        super(InceptionC, self).__init__()
        self.conv1 = nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(96)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.relu(out)

        return out

class Net_2D(nn.Module):
    def __init__(self):
        super(Net_2D, self).__init__()
        self.basic_conv = BasicConv()
        self.inception_a = InceptionA()
        self.inception_b = InceptionB()
        self.inception_c = InceptionC()

    def forward(self, x):
        out = self.basic_conv(x)
        out = self.inception_a(out)
        out = self.inception_b(out)
        out = self.inception_c(out)

        return out

class ResNet18_3D_3(nn.Module):
    def __init__(self):
        super(ResNet18_3D_3, self).__init__()
        self.conv1 = nn.Conv3d(96, 128, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm3d(128)
        self.conv2_1 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv2_2 = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv1(x)
        out = self.conv1_bn(residual)
        out = self.relu(out)
        out = self.conv2_1(out)
        out = self.conv2_1_bn(out)
        out = self.relu(out)
        out = self.conv2_2(out)

        out += residual

        out = self.conv2_bn(out)
        out = self.relu(out)

        return out

class ResNet18_3D_4(nn.Module):
    def __init__(self):
        super(ResNet18_3D_4, self).__init__()
        self.conv_down = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm3d(256)
        self.conv2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm3d(256)
        self.conv3 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm3d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv_down(x)

        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        residual = out

        out = self.conv2_bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = self.relu(out)
        out = self.conv4(out)

        out += residual

        out = self.conv4_bn(out)
        out = self.relu(out)

        return out

class ResNet18_3D_5(nn.Module):
    def __init__(self):
        super(ResNet18_3D_5, self).__init__()
        self.conv_down = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv1_bn = nn.BatchNorm3d(512)
        self.conv2 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm3d(512)
        self.conv3 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm3d(512)
        self.conv4 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm3d(512)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.conv_down(x)

        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual
        residual = out

        out = self.conv2_bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = self.relu(out)
        out = self.conv4(out)

        out += residual

        out = self.conv4_bn(out)
        out = self.relu(out)

        return out

class Net_3D(nn.Module):
    def __init__(self):
        super(Net_3D, self).__init__()
        self.res3 = ResNet18_3D_3()
        self.res4 = ResNet18_3D_4()
        self.res5 = ResNet18_3D_5()
        self.ga_pool = nn.AvgPool3d(kernel_size=5, stride=1, padding=0)

    def forward(self, x):
        out = self.res3(x)
        out = self.res4(out)
        out = self.res5(out)
        out = self.ga_pool(out)

        # テンソルサイズを変更
        # torch.Size([batch_num, 512, 1, 1, 1])からtorch.Size([batch_num, 512])へ
        out =out.view(out.size()[0], out.size()[1])

        return out

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

class ECO_Lite(nn.Module):
    name = 'eco_lite'
    is_replaced = True
    def __init__(self):
        super(ECO_Lite, self).__init__()
        self.net_2d = Net_2D()
        self.net_3d = Net_3D()
        self.classifier = nn.Sequential(
            DenseBlock(512+7, 256),
            nn.Linear(256, 101),
            nn.ReLU(inplace=True)
        )

    def forward(self, input, spec_tensor):
        bs, ch, x, y, z = input.shape
        out = input.permute(0, 4, 1, 2, 3)
        out = out.reshape(-1, ch, x, y)

        out = self.net_2d(out)
        out = out.view(-1, z, 96, 20, 20)
        out = out.permute(0, 2, 3, 4, 1)

        out = self.net_3d(out)

        # http://kaga100man.com/2019/03/25/post-102/
        out = torch.cat([out, spec_tensor], dim=1)

        out = self.classifier(out)

        return out