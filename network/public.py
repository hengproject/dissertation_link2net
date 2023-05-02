import functools
from torch import nn
from torch.nn import Identity
from torch.optim import lr_scheduler, Optimizer

from util.logger_util import get_logger


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    norm_layer = None
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = Identity()
    else:
        get_logger().error('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def _get_scheduler(optimizer, lr_policy, config):
    """Return a learning rate scheduler
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    log = get_logger()
    train_opt = config['train_opt']
    optim_opt = train_opt['optim']
    last_epoch = optim_opt['last_epoch']
    if lr_policy == 'linear':
        def lambda_rule(epoch):

            # lr_l = 1.0 - max(0, epoch - int(optim_opt['linear']['n_epoch_start_decay'])) / float(
            #     int(optim_opt['linear']['n_epoch_start_decay']) + 1)
            lr_l = 1.0 - max(0, epoch - int(optim_opt['linear']['n_epoch_start_decay'])) / float(
                int(optim_opt['linear']['n_decay_iter']) + 1)

            lr_l = max(lr_l, 1e-10)
            log.debug(f'scheduler linear reduce learning rate by {lr_l}')
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=optim_opt['step']['n_epoch_decay_frequency'],
                                        gamma=optim_opt['step']['step_rate'], last_epoch=last_epoch)
    elif lr_policy == 'plateau':
        plateau_opt = optim_opt['plateau']
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=float(plateau_opt['factor']),
                                                   threshold=float(plateau_opt['threshold']),
                                                   patience=int(plateau_opt['patience']),
                                                   cooldown=int(plateau_opt['cool_down']), eps=float(plateau_opt['eps'])
                                                   )
    elif lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_opt['train_n_epochs'],
                                                   eta_min=optim_opt['cosine']['eta_min'], last_epoch=last_epoch)
    else:
        log.error('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler


def get_schedulers(optimizer, config):
    lr_policy_list = config['train_opt']['optim']['policy']
    schedulers = [(_get_scheduler(optimizer, policy, config), policy) for policy in lr_policy_list]
    return schedulers


# update learning rate (called once every epoch)
def update_learning_rate(schedulers, optimizer, current_loss):
    for (scheduler, policy) in schedulers:
        if policy == 'plateau':
            scheduler.step(current_loss)
        else:
            scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    return lr



