import functools
import torch
from torch import nn


class NLayerDiscriminator(nn.Sequential):
    """Defines a PatchGAN discriminator"""
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        super().__init__()
        #  只有InstanceNorm2d才需要use_bias  BatchNorm2d已经做过偏置操作了
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kernel_size = 4
        padding = 1

        inConv = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding),
            nn.LeakyReLU(0.2, inplace = True)
        ]
        self.add_module("inConv",nn.Sequential(*inConv))

        # number of filter multiple
        filter_num_mult = 1
        # gradually increase the number of filters
        for n in range(1,n_layers):
            prev_filter_num_mult = filter_num_mult
            filter_num_mult = min( 2**n, 8 )
            nlayer_filter = [
                # input: 前一个filter的channel量 output: 新channel量 , 为2^n与8的最小值 like : 2 4 8 8 8 8 8
                nn.Conv2d(ndf * prev_filter_num_mult, ndf * filter_num_mult, kernel_size=kernel_size, stride=2, padding=padding, bias=use_bias),
                norm_layer(num_features=ndf * filter_num_mult),
                nn.LeakyReLU(0.2, True)
            ]
            self.add_module(f"nlayers_filter_{n-1}", nn.Sequential(*nlayer_filter))
        # 最后一个layer stride 为 1 ( WHY? )
        # 解答： https://zhuanlan.zhihu.com/p/359287990 此帖中指出，通过此类参数配置方式可以使得感受野为70*70，达到原论文中最好效果
        prev_filter_num_mult = filter_num_mult
        filter_num_mult = min(2 ** n_layers, 8)
        nlayer_filter = [
            # input: 前一个filter的channel量 output: 新channel量 , 为2^n与8的最小值 like : 2 4 8 8 8 8 8
            nn.Conv2d(ndf * prev_filter_num_mult, ndf * filter_num_mult, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=use_bias),
            norm_layer(num_features=ndf * filter_num_mult),
            nn.LeakyReLU(0.2, True)
        ]
        self.add_module(f"nlayers_filter_{n_layers}", nn.Sequential(*nlayer_filter))
        # output 1 channel as the prediction map
        output_layer = [
            nn.Conv2d(ndf * filter_num_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)
        ]
        self.add_module("output_layer",nn.Sequential(*output_layer))

    def forward(self, input):
        """Standard forward."""
        return super().forward(input)

if __name__ == "__main__":
    dummy_input = torch.ones((3, 3, 256, 256))
    resnet = NLayerDiscriminator(3, 3)
    print(resnet)
    result = resnet(dummy_input)
    print(result.shape)