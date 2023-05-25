import functools

from network.g_nets.link2net import Link2NetEncoder, Link2NetBlock
from network.g_nets.linknet import LinkEncoder, LinkDecoder
from util import logger_util
import torch
import torch.nn as nn
from network.g_nets.unet import UnetSkipConnectionBlock



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


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class LinkNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, gan_use=True):
        """Construct a Linknet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer  --9

        """
        super().__init__()
        InConv = [
            nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        self.inConv = nn.Sequential(*InConv)

        self.encoder1 = LinkEncoder(ngf, ngf, 3, 1, 1)
        self.encoder2 = LinkEncoder(ngf, ngf * 2, 3, 2, 1)
        self.encoder3 = LinkEncoder(ngf * 2, ngf * 4, 3, 2, 1)
        self.encoder4 = LinkEncoder(ngf * 4, ngf * 8, 3, 2, 1)

        self.decoder1 = LinkDecoder(ngf, ngf, 3, 1, 1, 0)
        self.decoder2 = LinkDecoder(ngf * 2, ngf, 3, 2, 1, 1)
        self.decoder3 = LinkDecoder(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.decoder4 = LinkDecoder(ngf * 8, ngf * 4, 3, 2, 1, 1)

        OutConv = [
            nn.ConvTranspose2d(ngf, ngf // 2, 3, 2, 1, 1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf // 2, ngf // 2, 3, 1, 1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf // 2, output_nc, 2, 2, 0, 0),
            nn.Tanh()
        ]
        self.outConv = nn.Sequential(*OutConv)

    def forward(self, input):
        # Initial block
        x = self.inConv(input)
        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        # Decoder blocks
        # d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)
        return self.outConv(d1)


class Link2NetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, layer_num=None, scales=4, base_ngf_multi=1,n_layer=0,leaky=False):
        """Construct a Linknet-based generator
            1m=26w in original paper
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer  --9
            layer_num ([int])   -- the number of res2net layers in the encoder
            scales (float|int)        -- the number of scales in the Res2Net block -- 4

        """
        super().__init__()
        if layer_num is None:
            layer_num = [3, 4, 6, 3]
        ngf = int(ngf * base_ngf_multi)
        InConv = [
            nn.Conv2d(in_channels=input_nc, out_channels=ngf, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ]
        self.inConv = nn.Sequential(*InConv)
        self.encoder1 = Link2NetEncoder(ngf, ngf, scales=scales,stride=1,n_blocks=layer_num[0])
        self.encoder2 = Link2NetEncoder(ngf, ngf * 2,scales=scales,stride=2,n_blocks=layer_num[1])
        self.encoder3 = Link2NetEncoder(ngf * 2, ngf * 4,scales=scales,stride=2,n_blocks=layer_num[2])
        self.encoder4 = Link2NetEncoder(ngf * 4, ngf * 8,scales=scales,stride=2,n_blocks=layer_num[3])

        blocks = [Link2NetBlock(ngf * 8,leaky=leaky) for _ in range(n_layer)]
        self.blocks = None
        if n_layer > 0:
            self.blocks = nn.Sequential(*blocks)

        self.decoder1 = LinkDecoder(ngf, ngf, 3, 1, 1, 0)
        self.decoder2 = LinkDecoder(ngf * 2, ngf, 3, 2, 1, 1)
        self.decoder3 = LinkDecoder(ngf * 4, ngf * 2, 3, 2, 1, 1)
        self.decoder4 = LinkDecoder(ngf * 8, ngf * 4, 3, 2, 1, 1)

        OutConv = [
            nn.ConvTranspose2d(ngf, ngf // 2, 3, 2, 1, 1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf // 2, ngf // 2, 3, 1, 1),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf // 2, output_nc, 2, 2, 0, 0),
            nn.Tanh()
        ]
        self.outConv = nn.Sequential(*OutConv)

    def forward(self, input):
        # Initial block
        x = self.inConv(input)
        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        if self.blocks is not None:
            e4 = e4 + self.blocks(e4)
        # Decoder blocks
        # d4 = e3 + self.decoder4(e4)
        d4 = e3 + self.decoder4(e4)
        d3 = e2 + self.decoder3(d4)
        d2 = e1 + self.decoder2(d3)
        d1 = x + self.decoder1(d2)
        return self.outConv(d1)


if __name__ == "__main__":
    from torchsummary import summary
    from util.time_util import time_start,time_end
    from thop import profile
    device = 'cuda'

    dummy_input = torch.ones((100,3, 256, 256)).to(device)
    u_net_gen = UnetGenerator(3, 3, 8).to(device)
    res_gen =  ResnetGenerator(3, 3).to(device)
    link_gen = LinkNetGenerator(3, 3).to(device)
    link2_gen =  Link2NetGenerator(3, 3, 64, layer_num=[3,4,6,3],scales=8,base_ngf_multi=1,n_layer=2).to(device)
   # gen = link_gen
    # result = gen(dummy_input)
    # print(result.shape)
    # print('link2gen')
    gen = link2_gen
    for i in range(20):
        start = time_start()
        result = gen(dummy_input)
        end = time_end(start)
        print(time_end(start))
    print(result.shape)
    # summary(gen,(3,256,256))
    flops, params = profile(gen, inputs=(dummy_input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    # summary(gen,(3,256,256))


