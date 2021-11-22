import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck

def resize(input, size=None, scale_factor=None, mode='nearest', align_corners=None, warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class stem(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super(stem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel*2, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel*2, track_running_stats=True),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.stem(x)
        output = self.maxpool(x)
        return output


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.stem = stem(in_channel=3, out_channel=32)
        self.layer1 = self._make_layer(planes=64, blocks=3)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6)
        self.layer4 = self._make_layer(planes=512, blocks=3)
        self.groups = 1
        self.base_width = 64

    def _make_layer(self, block=Bottleneck, planes=64, blocks=3, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                          padding=0, groups=1, bias=False, dilation=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, downsample=downsample, norm_layer=norm_layer, stride=stride)) # 较最开始的在这里加了Stride参数
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PPM(nn.ModuleList):

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    nn.Conv2d(self.in_channels, self.channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(self.channels, track_running_stats=True)
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:  # four scale deal respectively
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class PSPHead(nn.Module):
    """Pyramid Scene Parsing Network.
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), in_channels=2048, channels=512, align_corners=False):
        super(PSPHead, self).__init__()
        assert isinstance(pool_scales, (list, tuple))
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,  # 2048
            self.channels,  # 512
            align_corners=self.align_corners)
        self.bottleneck = nn.Sequential(# in_channel = 2048 + 4 * 512 = 4096)
            nn.Conv2d(self.in_channels + len(pool_scales) * self.channels,
                      self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels, track_running_stats=True)
        )

    def forward(self, inputs):
        """Forward function."""
        psp_outs = [inputs]
        psp_outs.extend(self.psp_modules(inputs))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output


class pspnet(nn.Module):
    def __init__(self):
        super(pspnet, self).__init__()
        self.backbone = ResNet50()
        self.decode_head = PSPHead()

    def forward(self, x):
        x = self.backbone(x)
        output = self.decode_head(x)
        return output


