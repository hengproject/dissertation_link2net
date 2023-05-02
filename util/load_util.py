import os.path

import torch
import yaml

from util.logger_util import get_logger


def get_device(config):
    device = None
    if config['use_cuda']:
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


def yaml_plain_loader(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    return result


"""
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr": lr
    }
"""


def load_checkpoint(checkpoint_file, model, optimizer, device):
    checkpoint_file = os.path.join(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr = checkpoint["lr"]

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_checkpoint_only_net(checkpoint_file, model, device):
    checkpoint_file = os.path.join(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])


if __name__ == "__main__":
    print(yaml_plain_loader("../config.yml"))
