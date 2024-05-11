import torch
import torch.nn as nn
from network.module import DoubleConv


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 逆卷积, kernel_size 和 stride 是相对于正卷积的过程而说的
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.up(x)  # 这一步就将 x 高宽加倍
        x = self.conv(x)  # 这里改变通道数量, 高宽不变

        return x
