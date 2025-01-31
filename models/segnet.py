import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, chin, chout, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(chin, chin, kernel_size=kernel_size, padding=kernel_size//2, groups=chin, bias=False),
            nn.Conv2d(chin, chout, kernel_size=1, bias=False),
            nn.BatchNorm2d(chout),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)

class ImageSegmentation(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.out_channels = 6
        self.bn_input = nn.BatchNorm2d(3)
        self.cd1 = DepthwiseSeparableConv(3, 64, kernel_size)
        self.dc1 = DownConv2(64, 64, kernel_size=kernel_size)
        self.cd2 = DepthwiseSeparableConv(64, 128, kernel_size)
        self.dc2 = DownConv2(128, 128, kernel_size=kernel_size)
        self.cd3 = DepthwiseSeparableConv(128, 256, kernel_size)
        self.dc3 = DownConv3(256, 256, kernel_size=kernel_size)
        self.cd4 = DepthwiseSeparableConv(256, 512, kernel_size)
        self.dc4 = DownConv3(512, 512, kernel_size=kernel_size)
        self.cd5 = DepthwiseSeparableConv(512, 1024, kernel_size)
        self.dc5 = DownConv3(1024, 1024, kernel_size=kernel_size)
        self.uc5 = UpConv3(1024, 512, kernel_size=kernel_size)
        self.cu1 = DepthwiseSeparableConv(1024, 512, kernel_size)
        self.uc4 = UpConv3(512, 256, kernel_size=kernel_size)
        self.cu2 = DepthwiseSeparableConv(512, 256, kernel_size)
        self.uc3 = UpConv3(256, 128, kernel_size=kernel_size)
        self.cu3 = DepthwiseSeparableConv(256, 128, kernel_size)
        self.uc2 = UpConv2(128, 64, kernel_size=kernel_size)
        self.cu4 = DepthwiseSeparableConv(128, 64, kernel_size)
        self.uc1 = UpConv2(64, self.out_channels, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.bn_input(batch)
        x1, mp1_indices, shape1 = self.dc1(self.cd1(x))
        x2, mp2_indices, shape2 = self.dc2(self.cd2(x1))
        x3, mp3_indices, shape3 = self.dc3(self.cd3(x2))
        x4, mp4_indices, shape4 = self.dc4(self.cd4(x3))
        x5, mp5_indices, shape5 = self.dc5(self.cd5(x4))
        x = self.uc5(x5, mp5_indices, output_size=shape5)
        x = torch.cat([x, x4], dim=1)
        x = self.cu1(x)
        x = self.uc4(x, mp4_indices, output_size=shape4)
        x = torch.cat([x, x3], dim=1)
        x = self.cu2(x)
        x = self.uc3(x, mp3_indices, output_size=shape3)
        x = torch.cat([x, x2], dim=1)
        x = self.cu3(x)
        x = self.uc2(x, mp2_indices, output_size=shape2)
        x = torch.cat([x, x1], dim=1)
        x = self.cu4(x)
        x = self.uc1(x, mp1_indices, output_size=shape1)
        return x