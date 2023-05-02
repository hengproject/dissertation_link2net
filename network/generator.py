import functools
from util import logger_util
import torch
import torch.nn as nn


class ResnetGenerator(nn.Sequential):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', leaky_relu=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer  --9
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super().__init__()
        #  只有InstanceNorm2d才需要use_bias  BatchNorm2d已经做过偏置操作了
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # inconv
        InConv = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(ngf),
                  nn.ReLU(True)]
        self.add_module("inConv", nn.Sequential(*InConv))
        n_upsampling = 2
        n_downsampling = 2
        # 做两次down sample 第一次(ngf,ngf*2) 第二次(ngf*2,ngf*4)
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            down = Down(ngf * mult, ngf * mult * 2, norm_layer, use_bias, leaky_relu)
            self.add_module(f"down_sample_{i}", down)
        # 增加resblock , resblock 的channel数与最后一次 downsample 的out_channel一样
        mult = 2 ** n_downsampling
        resblocks = []
        for i in range(n_blocks):  # add ResNet blocks
            resblocks += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        self.add_module("resblocks", nn.Sequential(*resblocks))
        # 开始upsample
        for i in range(n_upsampling):
            mult = 2 ** (n_downsampling - i)
            up = Up(ngf * mult, int(ngf * mult / 2), norm_layer, use_bias, leaky_relu)
            self.add_module(f"up_sample_{i}", up)
        # 利用conv 转换成 output所需要的channel数量
        output_conv = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.add_module("output_conv", nn.Sequential(*output_conv))

    def forward(self, input):
        return super().forward(input)


class Down(nn.Module):
    """
        down sampling module
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_bias, leaky_relu=False):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class Up(nn.Module):
    """
        up sampling module
    """

    def __init__(self, in_channel, out_channel, norm_layer, use_bias, leaky_relu=False):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            norm_layer(out_channel),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        # 根据padding_type，如果为zero则试用Conv2d默认模块进行padding 否则加入 get_padding_layer_by_type 作为padding
        p = 0
        if padding_type == 'zero':
            p = 1
        else:
            conv_block.append(self.get_padding_layer_by_type(padding_type))

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'zero':
            p = 1
        else:
            conv_block.append(self.get_padding_layer_by_type(padding_type))
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    @staticmethod
    def get_padding_layer_by_type(padding_type):
        if padding_type == 'reflect':
            return nn.ReflectionPad2d(1)
        elif padding_type == 'replicate':
            return nn.ReplicationPad2d(1)
        else:
            logger_util.get_logger().error(f'padding [[{padding_type}]] is not implemented')

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


if __name__ == "__main__":
    dummy_input = torch.ones((1, 3, 256, 256))
    resnet = ResnetGenerator(3, 3)
    result = resnet(dummy_input)
    print(result.shape)
    # print(resnet)
