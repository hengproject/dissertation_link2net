import torch
import torch.nn as nn
# from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import itertools


class Tmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass


# def test1():
#     decay_epoch = 100
#     model = Tmodel()
#
#     def lambda_rule(epoch):
#         # lr_l = 1.0 - max(0, epoch - int(optim_opt['linear']['n_epoch_start_decay'])) / float(
#         #     int(optim_opt['linear']['n_epoch_start_decay']) + 1)
#         lr_l = 1.0 - max(0, epoch - 50) / 101.0
#
#         lr_l = max(lr_l, 1e-10)
#         return lr_l
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
#
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=-1)
#
#     print("初始化的学习率：", optimizer.defaults['lr'])
#
#     lr_list = []  # 把使用过的lr都保存下来，之后画出它的变化
#
#     for epoch in range(1, 400):
#         # train
#         optimizer.zero_grad()
#         optimizer.step()
#         print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
#         lr_list.append(optimizer.param_groups[0]['lr'])
#         scheduler.step()
#
#     # 画出lr的变化
#     plt.plot(list(range(1, 400)), lr_list)
#     plt.xlabel("epoch")
#     plt.ylabel("lr")
#     plt.title("learning rate's curve changes as epoch goes on!")
#     plt.show()
#
#
#
# def test2():
#
#     model = Tmodel()
#
#     optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
#     scheduler = CosineAnnealingLR(optimizer, T_max=200,eta_min=0)
#
#     print("初始化的学习率：", optimizer.defaults['lr'])
#
#     lr_list = []  # 把使用过的lr都保存下来，之后画出它的变化
#
#     for epoch in range(1, 400):
#         # train
#         optimizer.zero_grad()
#         optimizer.step()
#         print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
#         lr_list.append(optimizer.param_groups[0]['lr'])
#         scheduler.step()
#
#     # 画出lr的变化
#     plt.plot(list(range(1, 400)), lr_list)
#     plt.xlabel("epoch")
#     plt.ylabel("lr")
#     plt.title("learning rate's curve changes as epoch goes on!")
#     plt.show()


def get_kwarg(**kwargs):
    return kwargs

if __name__ == '__main__':
    print(get_kwarg(save_folder=1, epoch=2, gen=3, disc=4,
                             gen_optim=5, disc_optim=66, g_lr=7, d_lr=8, save_opt=8,
                             test_dataloader=8, config=9,device=9))



