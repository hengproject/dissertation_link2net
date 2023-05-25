import torch
from torch import nn


class Conv1x1BN(nn.Sequential):
    def __init__(self, input_nc, output_nc, stride=1, padding=0,groups=1):
        super().__init__()
        self.add_module("Conv1x1BN", nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=stride, padding=padding,groups=groups),
            nn.BatchNorm2d(output_nc)
        ))

    def forward(self, input):
        return super().forward(input)


class Conv3x3BN(nn.Sequential):
    def __init__(self, input_nc, output_nc, stride=1, padding=0, groups=1):
        super().__init__()
        self.add_module("Conv3x3BN", nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(output_nc)
        ))

    def forward(self, input):
        return super().forward(input)


class Conv5x5BN(nn.Sequential):
    def __init__(self, input_nc, output_nc, stride=1, padding=0, groups=1):
        super().__init__()
        self.add_module("Conv5x5BN", nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=5, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(output_nc)
        ))

    def forward(self, input):
        return super().forward(input)


class Conv7x7BN(nn.Sequential):
    def __init__(self, input_nc, output_nc, stride=1, padding=0, groups=1):
        super().__init__()
        self.add_module("Conv7x7BN", nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=7, stride=stride, padding=padding, groups=groups),
            nn.BatchNorm2d(output_nc)
        ))

    def forward(self, input):
        return super().forward(input)


class Conv1x1BNRelu(Conv1x1BN):
    def __init__(self, input_nc, output_nc, stride=1, padding=0, groups=1):
        super().__init__(input_nc, output_nc, stride, padding, groups)

    def forward(self, x):
        x = super().forward(x)
        return nn.ReLU(inplace=True)(x)

class Conv3x3BNRelu(Conv3x3BN):
    def __init__(self, input_nc, output_nc, stride=1, padding=0, groups=1):
        super().__init__(input_nc, output_nc, stride, padding, groups)

    def forward(self, x):
        x = super().forward(x)
        return nn.ReLU(inplace=True)(x)

class Conv5x5BNRelu(Conv5x5BN):
    def __init__(self, input_nc, output_nc, stride=1, padding=0, groups=1):
        super().__init__(input_nc, output_nc, stride, padding, groups)

    def forward(self, x):
        x = super().forward(x)
        return nn.ReLU(inplace=True)(x)

class Conv7x7BNRelu(Conv7x7BN):
    def __init__(self, input_nc, output_nc, stride=1, padding=0, groups=1):
        super().__init__(input_nc, output_nc, stride, padding, groups)

    def forward(self, x):
        x = super().forward(x)
        return nn.ReLU(inplace=True)(x)
# SE模块
class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


"""
    output_nc = L*scale*expansion (16K) 
"""


class Res2NetBottleneck(nn.Module):

    def __init__(self, input_nc, output_nc, stride=1, scales=4, groups=1, se=True, expansion=4):
        super(Res2NetBottleneck, self).__init__()
        # 残差块的输出通道数=输入通道数*expansion
        # scales为残差块中使用分层的特征组数，groups表示其中3*3卷积层数量，SE模块和BN层
        # output_nc = expansion * plane
        # plane = k * scale
        # output_nc = K * scale * expansion
        self.expansion = expansion
        if output_nc % (scales * expansion) != 0:  # 输出通道数为4的倍数
            raise ValueError(f'output_nc({output_nc}) must be divisible by scales * expansion({scales * expansion})')
        self.scales = scales
        self.downsample = None
        if stride != 1 or input_nc != (output_nc // self.expansion):
            self.downsample = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=stride),
                nn.BatchNorm2d(output_nc)
            )
        bottleneck_channels = groups * (output_nc // self.expansion)
        self.models = nn.ModuleDict({
            '1x1ConvBNRelu_in': Conv1x1BNRelu(input_nc, bottleneck_channels, stride),
            '1x1ConvBNRelu_out': Conv1x1BNRelu(bottleneck_channels, output_nc, stride=1),
            # 3*3的卷积层，一共有scale-1个
            '3x3ConvBNList': nn.ModuleList(
                [Conv3x3BNRelu(bottleneck_channels // scales, bottleneck_channels // scales, stride=1, padding=1,
                               groups=groups)
                 for _ in range(scales - 1)
                 ]
            )
        })
        self.se = SEModule(output_nc) if se else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.models['1x1ConvBNRelu_in'](x)
        # scales个(3x3)的残差分层架构
        xs = torch.chunk(out, self.scales, dim=1)
        ys = []
        for slice in range(self.scales):
            if slice == 0:
                ys.append(xs[slice])
            elif slice == 1:
                out = self.models['3x3ConvBNList'][slice - 1](xs[slice])
                ys.append(out)
            else:
                out = self.models['3x3ConvBNList'][slice - 1](xs[slice] + ys[-1])
                ys.append(out)

        out = torch.cat(ys, dim=1)
        out = self.models['1x1ConvBNRelu_out'](out)

        # 加入SE模块
        if self.se is not None:
            out = self.se(out)
        # 下采样
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


if __name__ == '__main__':
    channel = 3
    # dummy_input = torch.ones((9, channel, 256, 256))
    # conv = nn.Conv2d(channel, 32, kernel_size=1, stride=2,groups=2)
    # print(conv(dummy_input).shape)

    dummy_input = torch.ones((1, 16, 256, 256))
    # model = Res2Net( [3, 4, 23, 3], width=26, scales=4,num_classes=3)
    stride = 2
    model = Conv1x1BNRelu(input_nc=16, output_nc=8, stride=2)
    print(model(dummy_input).shape)
