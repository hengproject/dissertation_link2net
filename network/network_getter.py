import torch
from torch.nn import init

from .discriminator import NLayerDiscriminator
from .generator import ResnetGenerator, UnetGenerator, LinkNetGenerator, Link2NetGenerator
from .public import get_norm_layer
from util.logger_util import get_logger

def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    elif netG == 'unet':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif netG == 'linknet':
        net = LinkNetGenerator(input_nc, output_nc, ngf)
    elif netG == 'link2net50_1m_8s':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3,4,6,3],scales=8,base_ngf_multi=1)
    elif netG == 'link2net50_1m_16s':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3,4,6,3],scales=16,base_ngf_multi=1)
    elif netG == 'link2net50_1.5m_2s':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3,4,6,3],scales=2,base_ngf_multi=1.5)
    elif netG == 'link2net50_1.5m_8s':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3,4,6,3],scales=8,base_ngf_multi=1.5)
    elif netG == 'link2net50_2m_16s':
        net = Link2NetGenerator(3, 3, layer_num=[3, 4, 6, 3], scales=16, base_ngf_multi=2)
    elif netG == 'link2net50_4m_16s':
        net = Link2NetGenerator(3, 3, layer_num=[3, 4, 6, 3], scales=16, base_ngf_multi=4)
    elif netG == 'link2net101_1m_4s':
        net = Link2NetGenerator(input_nc, output_nc, ngf,scales=4,layer_num=[3, 4, 23, 3])
    elif netG == 'link2net50_1m_8s_2block':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3,4,6,3],scales=8,base_ngf_multi=1,n_layer=2)
    elif netG == 'link2net50_1m_8s_4block':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3, 4, 6, 3], scales=8, base_ngf_multi=1, n_layer=4)
    elif netG == 'link2net50_1m_8s_6block':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3,4,6,3],scales=8,base_ngf_multi=1,n_layer=6)
    elif netG == 'link2net50_1m_4s_6block':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3, 4, 6, 3], scales=4, base_ngf_multi=1, n_layer=6)
    elif netG == 'link2net50_1m_4s_8block':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3, 4, 6, 3], scales=4, base_ngf_multi=1, n_layer=8)
    elif netG == 'link2net50_1.5m_8s_9block':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3,4,6,3],scales=8,base_ngf_multi=1.5,n_layer=9)
    elif netG == 'link2net50_1m_16s_2block':
        net = Link2NetGenerator(input_nc, output_nc, ngf, layer_num=[3,4,6,3],scales=16,base_ngf_multi=1,n_layer=2)

    else:
        get_logger().error('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_id)

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    # elif netD == 'pixel':     # classify if each pixel is real or fake
    #     net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
    else:
        get_logger().error('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_id)



def init_net(net:torch.nn.Module, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_id == 'cpu':
        net.to('cpu')
    else:
        if torch.cuda.is_available():
            net.to(gpu_id)
        else:
            get_logger().error("cuda is not available")
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>