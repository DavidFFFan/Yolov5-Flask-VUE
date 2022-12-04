# -*-coding:utf-8-*-
import torch.nn as nn
import torch


# ‘’‘
# 该函数分别进行3x3卷积 BN ReLU6操作
# ’‘’
def conv2d_bn_relu6(in_channels, out_channels, kernel_size=3, stride=2, dropout_prob=0.0):
    # To preserve the equation of padding. (k=1 maps to pad 0, k=3 maps to pad 1, k=5 maps to pad 2, etc.)
    padding = (kernel_size + 1) // 2 - 1
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        # For efficiency, Dropout is placed before Relu.
        nn.Dropout2d(dropout_prob, inplace=True),
        # Assumption: Relu6 is used everywhere.
        nn.ReLU6(inplace=True)
    )

class Encoder(nn.Module):
    def __init__(self, ngf=64):
        super(Encoder, self).__init__()
        self.con1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.LeakyReLU(0.2))

        self.con2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf*2, affine=True),
            nn.LeakyReLU(0.2))

        self.con3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf * 4, affine=True),
            nn.LeakyReLU(0.2))

        self.con4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 8, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf * 8, affine=True),
            nn.LeakyReLU(0.2))

        self.con5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(ngf * 8, affine=True),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        con1 = self.con1(x)
        con2 = self.con2(con1)
        con3 = self.con3(con2)
        con4 = self.con4(con3)
        con5 = self.con5(con4)
        return [con1, con2, con3, con4, con5]


class Decoder(nn.Module):
    def __init__(self, ngf=64): # 很奇怪，为什么这里ngf=32？
        super(Decoder, self).__init__()
        self.decon4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf * 8, affine=True),
            nn.ReLU(inplace=True))

        self.decon5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 8 * 2, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf * 4, out_channels=ngf * 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf * 4, affine=True),
            nn.ReLU(inplace=True))

        self.decon6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 2, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf * 2, affine=True),
            nn.ReLU(inplace=True))

        self.decon7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 4, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True))

        self.decon8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(ngf, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=3, stride=1, padding=0),
            nn.Tanh())

        self.encoder = Encoder(ngf=ngf)

    def forward(self, x):
        con1, con2, con3, con4, con5 = self.encoder(x)
        decon4 = self.decon4(con5)
        decon4 = torch.cat([decon4, con4], dim=1)

        decon5 = self.decon5(decon4)
        decon5 = torch.cat([decon5, con3], dim=1)

        decon6 = self.decon6(decon5)
        decon6 = torch.cat([decon6, con2], dim=1)

        decon7 = self.decon7(decon6)
        decon7 = torch.cat([decon7, con1], dim=1)

        decon8 = self.decon8(decon7)
        return decon8


class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=3*2, out_channels=ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 2, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 4, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 8, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator_UP(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator_UP, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=3, out_channels=ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 2, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 4, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * 8, affine=True))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    decoder = Decoder(64)

