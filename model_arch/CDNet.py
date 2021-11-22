import torch.nn as nn

class ConvBlock(nn.Module):
    """double conv layers block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DconvBlock(nn.Module):
    """Downscale block: maxpool -> double conv block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Dconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.Dconv(x)


class OutputBlock(nn.Module):
    """Output block: double conv block -> output conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.out_conv(x)


class CDNet(nn.Module):
    def __init__(self):
        super(CDNet, self).__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.dconv1 = DconvBlock(64, 32)
        self.dconv2 = DconvBlock(32, 64)
        self.dconv3 = DconvBlock(64, 32)
        self.Output = OutputBlock(32, 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.dconv1(x2)
        x4 = self.dconv2(x3)
        x5 = self.dconv3(x4)
        out = self.Output(x5)
        return out

