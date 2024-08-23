import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.nn import Conv2d, LeakyReLU, MaxPool2d, Flatten, Linear, Dropout, Upsample, ReLU, ConvTranspose2d, Tanh, BatchNorm2d
from torchsummary import summary # type: ignore
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter


KERNEL_SIZE = 2
STRIDE = 2

class GeneratorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2d(1, 64, kernel_size=2, stride=2)
        self.db1 = nn.Sequential(
            LeakyReLU(),
            Conv2d(64, 128, kernel_size=2, stride=2),
            BatchNorm2d(128)
        )
        self.db2 = nn.Sequential(
            LeakyReLU(),
            Conv2d(128, 1256, kernel_size=2, stride=2),
            BatchNorm2d(1256)
        )
        self.db3 = nn.Sequential(
            LeakyReLU(),
            Conv2d(1256, 512, kernel_size=2, stride=2),
            BatchNorm2d(512)
        )
        self.db4 = nn.Sequential(
            LeakyReLU(),
            Conv2d(512, 512, kernel_size=2, stride=2),
            BatchNorm2d(512)
        ) ## same as db 5 and db6
        self.db5 = nn.Sequential(
            LeakyReLU(),
            Conv2d(512, 512, kernel_size=2, stride=2),
            BatchNorm2d(512)
        )
        self.db6 = nn.Sequential(
            LeakyReLU(),
            Conv2d(512, 512, kernel_size=2, stride=2),
            BatchNorm2d(512)
        )
        self.dl = nn.Sequential(
            LeakyReLU(),
            Conv2d(512, 512, kernel_size=2, stride=2),
        )

        self.uf = nn.Sequential(
            ReLU(),
            ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            BatchNorm2d(512)
        )
        self.ub1 = nn.Sequential(
            ReLU(),
            ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            BatchNorm2d(512)
        ) ## same as ub2 and ub3
        self.ub2 = nn.Sequential(
            ReLU(),
            ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            BatchNorm2d(512)
        )
        self.ub3 = nn.Sequential(
            ReLU(),
            ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            BatchNorm2d(512)
        )
        self.ub4 = nn.Sequential(
            ReLU(),
            ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            BatchNorm2d(256)
        )
        self.ub5 = nn.Sequential(
            ReLU(),
            ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            BatchNorm2d(128)
        )

        self.ub6 = nn.Sequential(
            ReLU(),
            ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            BatchNorm2d(64)
        )
        self.ul = nn.Sequential(
            ReLU(),
            ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            Tanh()
        )

    def forward(self, x):
        c1 = self.conv1(x)
        # print(f"c1 shape is {c1.shape}")
        d1 = self.db1(c1)
        # print(f"d1 shape is {d1.shape}")
        d2 = self.db2(d1)
        # print(f"d2 shape is {d2.shape}")
        d3 = self.db3(d2)
        # print(f"d3 shape is {d3.shape}")
        d4 = self.db4(d3)
        # print(f"d4 shape is {d4.shape}")
        d5 = self.db5(d4)
        # print(f"d5 shape is {d5.shape}")
        d6 = self.db6(d5)
        # print(f"d6 shape is {d6.shape}")
        l = self.dl(d6)
        # print(f"l shape is {l.shape}")
        # g = self.generator(l)
        f = self.uf(l)
        # print(f"f shape is {f.shape}")
        conv = Conv2d(1024, 512, kernel_size=1).to(device=device)
        u1 = self.ub1(conv(torch.cat([f, d6], dim=1)))
        # print(f"u1 shape is {u1.shape}")
        u2 = self.ub2(conv(torch.concat([u1, d5], dim=1)))
        # print(f"u2 shape is {u2.shape}")
        u3 = self.ub3(conv(torch.concat([u2, d4], dim=1)))
        # print(f"u3 shape is {u3.shape}")
        u4 = self.ub4(conv(torch.concat([u3, d3], dim=1)))
        # print(f"u4 shape is {u4.shape}")
        conv2 = Conv2d(1512, 256, kernel_size=1).to(device=device)
        u5 = self.ub5(conv2(torch.concat([u4, d2], dim=1)))
        # print(f"u5 shape is {u5.shape}")
        conv3 = Conv2d(256, 128, kernel_size=1).to(device=device)
        u6 = self.ub6(conv3(torch.concat([u5, d1], dim=1)))
        # print(f"u6 shape is {u6.shape}")
        c1_cropped = c1[:, :, :510, :510]
        conv4 = Conv2d(128, 64, kernel_size=1).to(device=device)
        u7 = self.ul(conv4(torch.concat([u6, c1_cropped], dim=1)))
        # print(f"u7 shape is {u7.shape}")
        return u7
        

class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.bl0 = nn.Sequential(
            Conv2d(4, 64, kernel_size=4, stride=2),
            LeakyReLU(),
        )
        self.bl1 = nn.Sequential(
            Conv2d(64, 128, kernel_size=4, stride=2),
            BatchNorm2d(128),
            LeakyReLU()
        )
        self.bl2 = nn.Sequential(
            Conv2d(128, 256, kernel_size=4, stride=2),
            BatchNorm2d(256),
            LeakyReLU()
        )
        self.bl3 = nn.Sequential(
            Conv2d(256, 512, kernel_size=4, stride=2),
            BatchNorm2d(512),
            LeakyReLU()
        )

        self.conv = Conv2d(512, 1, kernel_size=4, stride=2)
        
    def forward(self, x):
        # x = torch.cat([SAR, cSAR], dim=1)
        x = self.bl0(x)
        x = self.bl1(x)
        x = self.bl2(x)
        x = self.bl3(x)
        x = self.conv(x)
        return x




class SARModel():
    def __init__(self):
        self.generator = GeneratorModel().to(device=device)
        self.discriminator = DiscriminatorModel().to(device=device)
        self.writer = SummaryWriter('runs/sar')

    def train(self, train_dataloader:DataLoader, test_dataloader:DataLoader, epochs:int = 10):
        pass


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    g_model = GeneratorModel().to(device=device)
    print(summary(g_model, (1, 256, 256)))
    # print(summary(g_model, (1, 1024, 1024)))
    # print(g_model(torch.randn(1, 1, 1024, 1024).to(device=device)))

    d_model = DiscriminatorModel().to(device=device)
    print(summary(d_model, (4, 256, 256)))