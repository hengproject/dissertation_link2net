import torch
from torch import nn
from util.logger_util import get_logger
from util.load_util import yaml_plain_loader as config_loader
from util.test_util import SSIM, MS_SSIM


class GANLoss(nn.Module):
    """Define different GAN objectives.

       The GANLoss class abstracts away the need to create the target label tensor
       that has the same size as the input.
       """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            get_logger().error('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss


class MixedIndicationLoss(nn.Module):
    """
    define the Loss from several indications,like ssim,mae(L1)
    """

    def __init__(self, g_loss,device):
        super().__init__()
        self.l1 = None
        self.l1_func = nn.L1Loss()
        self.ssim = None
        self.ms_ssim = None
        self.sequence = []
        self.device = device
        for each in g_loss:
            name = each['name']
            weight = float(each['weight'])
            if name == 'l1':
                self.sequence.append({"func": self.l1_loss, 'weight': weight})
            elif name == 'ssim':
                self.sequence.append({"func": self.ssim_loss, 'weight': weight})
            elif name == 'ms_ssim':
                self.sequence.append({"func": self.ms_ssim_loss, 'weight': weight})

    def l1_loss(self,X,Y):
        self.l1 = self.l1_func(X,Y)
        return self.l1

    def ssim_loss(self, X, Y):
        self.ssim = SSIM(X, Y, data_range=1)
        return 1 - self.ssim

    def ms_ssim_loss(self, X, Y):
        self.ms_ssim = MS_SSIM(X, Y, data_range=1)
        return 1 - self.ms_ssim

    def __call__(self, tensor_a, tensor_b):
        total_loss = torch.zeros(1).to(self.device)
        for each in self.sequence:
            total_loss += each['func'].__call__(tensor_a, tensor_b) * each['weight']
        return total_loss


if __name__ == '__main__':
    device = 'cuda:0'
    config = config_loader('../config_linux.yml')
    train_opt = config['train_opt']
    criterionGLoss = MixedIndicationLoss(train_opt['loss']['g_loss']).to(device)
    torch.manual_seed(123)
    tensor_a_ = torch.randn((5, 256, 256, 256))
    tensor_b_ = torch.randn((5, 256, 256, 256))
    criterion_gloss = criterionGLoss(tensor_a_, tensor_b_)
    print(criterion_gloss)
    print(criterionGLoss.ssim)
    print(criterionGLoss.ms_ssim)
    print(criterionGLoss.l1)
