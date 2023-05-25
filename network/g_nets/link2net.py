import torch
from torch import nn
from network.g_nets.res2net import Res2NetBottleneck, Conv1x1BN, Conv1x1BNRelu, Conv3x3BN


class Link2NetEncoder(nn.Module):
    def __init__(self, in_planes, out_planes,stride,scales,n_blocks=3):
        super().__init__()
        block1 = Res2NetBottleneck(in_planes, out_planes,stride=stride,scales=scales)
        block2list = [
            Res2NetBottleneck(out_planes, out_planes,stride=1,scales=scales)
            for _ in range(1,n_blocks)
            ]
        self.encoder = nn.Sequential(
            block1, *block2list
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Link2NetBlock(nn.Module):
    def __init__(self,dim,leaky=False):
        super().__init__()
        self.conv1 = Conv1x1BNRelu(dim,dim)
        self.conv2 = nn.Sequential(
            Conv1x1BN(dim, dim),
            nn.Conv2d(dim,dim,kernel_size=(1,3),padding=1),
            nn.BatchNorm2d(dim),
            self.get_relu(leaky),
            nn.Conv2d(dim,dim,kernel_size=(3,1),padding=0),
            nn.BatchNorm2d(dim),
            self.get_relu(leaky)
        )
        self.conv3 = nn.Sequential(
            Conv1x1BN(dim, dim),
            Conv3x3BN(dim,dim,padding=1),
            self.get_relu(leaky)
        )
        self.outConv = nn.Sequential(
            Conv1x1BN(dim*3,dim*1),
            self.get_relu(leaky)
        )
    def get_relu(self,leaky):
        return nn.LeakyReLU(1e-2) if leaky else nn.ReLU(inplace=True)

    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        out = torch.cat([x1,x2,x3],dim=1)
        out = self.outConv(out)+x
        return out

if __name__ == '__main__':
    dummy_input = torch.ones((1,3, 256, 256))
    encoder = Link2NetEncoder(3, 64, stride=2,scales=8)
    print(encoder(dummy_input).shape)
