import os

import torch
import yaml
from datetime import datetime

from util.logger_util import get_logger
from util.visualize_util import tensor_to_PIL


def _init_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# all the log_data and model_data will be saved in a sub-folder contained in the root-path dir path
def init_save(root_path, config, sub_dir_name=None):
    _init_dir(root_path)
    now = datetime.now()
    if sub_dir_name is None:
        sub_dir_name = now.strftime("%Y-%m-%d--%H-%M-%S")
    sub_dir_path = os.path.join(root_path, sub_dir_name)
    _init_dir(sub_dir_path)
    save_yaml(os.path.join(sub_dir_path, "train_config.yaml"), config)
    return sub_dir_path

def init_test_save(root_path):
    test_path = os.path.join(root_path,'test')
    _init_dir(test_path)
    return test_path



def save_yaml(path, obj):
    with open(path, 'w') as f:
        yaml.dump(obj, f)


def save_model(file_name, model, optimizer, lr):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr": lr
    }
    torch.save(checkpoint, file_name)


def save_gen_model(epoch_folder, epoch, gen, optimizer, lr):
    file_name = os.path.join(epoch_folder, r"gen_model.pth")
    save_model(file_name, model=gen, optimizer=optimizer, lr=lr)
    get_logger().info(f"epoch[[{epoch}]] generator model saved to {file_name}")
    return file_name


def save_disc_model(epoch_folder, epoch, disc, optimizer, lr):
    file_name = os.path.join(epoch_folder, r"disc_model.pth")
    save_model(file_name, model=disc, optimizer=optimizer, lr=lr)
    get_logger().info(f"epoch[[{epoch}]] discriminator model saved to {file_name}")
    return file_name


def save_some_example(num, gen, test_dataloader,device, epoch_dir=None ,save=True,ret_result=False):
    ret = []
    with torch.no_grad():
        gen.eval()
        # batch[0] (1,3,256,256)
        for i, batch in enumerate(test_dataloader):
            real_a = batch[0].to(device)
            fake_b = gen(real_a)
            real_a = _reverse_normalization(batch[0][0]).cpu()
            real_b = _reverse_normalization(batch[1][0]).cpu()
            fake_b = _reverse_normalization(fake_b[0]).cpu()
            all = torch.cat([real_a, real_b, fake_b], dim=2)
            if save:
                _init_dir(epoch_dir)
                save_pil_image(tensor_to_PIL(all), path=os.path.join(epoch_dir, f'{i}.jpeg'))
            if ret_result:
                ret.append(all.cpu())
            if i == num-1: break
        gen.train()
        return ret


def save_pil_image(pil, path):
    pil.save(path)


def _reverse_normalization(tensor):
    return tensor * 0.5 + 0.5


def check_point_save(save_folder, epoch, gen, disc, gen_optim, disc_optim, g_lr, d_lr, save_opt, test_dataloader,
                     config,device):
    config['train_opt']["starting_epoch"] = epoch
    config['train_opt']['optim']['last_epoch'] = epoch
    config['train_opt']['optim']['g_learning_rate'] = g_lr
    config['train_opt']['optim']['d_learning_rate'] = d_lr
    config['train_opt']['continue_train_opt']['activate'] = True
    example_opt = save_opt['save_model']['save_example']
    epoch_folder = os.path.join(save_folder, str(epoch))
    _init_dir(epoch_folder)
    if example_opt['activate']:
        save_some_example(example_opt['num'], gen, test_dataloader,device, epoch_folder,)
    g_path = save_gen_model(epoch_folder, epoch, gen, gen_optim, g_lr)
    d_path = save_disc_model(epoch_folder, epoch, disc, disc_optim, d_lr)
    config['train_opt']['continue_train_opt']['g_model_path'] = g_path
    config['train_opt']['continue_train_opt']['d_model_path'] = d_path
    save_yaml(os.path.join(epoch_folder, 'continue_train.yaml'), config)
