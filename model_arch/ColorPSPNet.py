import torch
import torch.nn as nn
from model_arch.CBAM_blocks import CBAM
from model_arch.pspnet import pspnet

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return torch.relu(self.conv(x))


class DownBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            DoubleConvBlock(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.downsample(x)


class BridgeUP(nn.Module):
    """Downscale bottleneck block: conv -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_up = nn.Sequential(
            DoubleConvBlock(in_channels, in_channels),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_up(x)


class UpBlock(nn.Module):
    """Upscale block: double conv block -> transpose conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels, in_channels)
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return torch.relu(self.up(x))


class ColorPSPNet(nn.Module):
    def __init__(self):
        super(ColorPSPNet, self).__init__()
        self.pspnet = pspnet()
        self.conv1 = DownBlock(3, 32)
        self.conv2 = DownBlock(32, 64)
        self.conv3 = DownBlock(64, 128)

        self.dconv1 = BridgeUP(256, 64)
        self.dconv2 = UpBlock(128, 32)
        self.dconv3 = UpBlock(64, 32)
        self.output = nn.Sequential(
            nn.Conv2d( 32, 3, kernel_size=3, padding=1), 
            nn.Conv2d( 3, 3, kernel_size=3, padding=1)
        )

        self.CBAM1 = CBAM(32)
        self.CBAM2 = CBAM(64)
        self.CBAM3 = CBAM(128)

        self.connect1 = nn.Sequential( DoubleConvBlock(32, 32), DoubleConvBlock(32, 32) )
        self.connect2 = DoubleConvBlock(64, 64)
        self.connect_sem = nn.Sequential(
            DoubleConvBlock(512, 128),   
            DoubleConvBlock(128, 128)
        )

        self.pspnet.load_state_dict(torch.load('../models/pspnet_state.pth'))
        for param in self.pspnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x_cg_0 = self.pspnet(x)
        x_cg_1 = self.connect_sem(x_cg_0)
        x_cg_1 = self.CBAM3(x_cg_1)

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x_1 = self.connect1(x1)
        x_1 = self.CBAM1(x_1)

        x_2 = self.connect2(x2)
        x_2 = self.CBAM2(x_2)

        x3_1 = torch.cat((x3, x_cg_1), dim=1)
        x4 = self.dconv1(x3_1)
        x5 = self.dconv2(x4, x_2)
        x6 = self.dconv3(x5, x_1)
        out = self.output(x6)
        return out

if __name__ == '__main__':
    net = ColorPSPNet()
    print(net)