import torch
import torch.nn as nn
from bluenet2.module import DoubleConv
from bluenet2.module import Attention


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        # 逆卷积, kernel_size 和 stride 是相对于正卷积的过程而说的
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.conv = DoubleConv(in_channels, out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        # 添加一层 Attention 上采样一次就做一次特征融合
        self.attn = Attention(
            dim=out_channels,  # channel 就相当于对每个像素点的编码长度
            num_heads=4,
            qkv_bias=False,
            qk_scale=None,
            sr_ratio=reduction
        )
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.up(x)  # 这一步就将 x 高宽加倍
        x = self.norm1(x)
        x = self.conv(x)  # 这里改变通道数量, 高宽不变
        x = self.norm2(x)
        r = x
        x = self.attn(x)
        x = x + r  # 做残差
        self.norm3(x)

        return x


if __name__ == '__main__':
    x1 = torch.rand(size=(8, 8, 224, 224))
    x2 = torch.rand(size=(8, 8, 224, 224))

    up = Up(in_channels=16, out_channels=32, reduction=64)
    outputs = up(x1, x2)
    print(f'x1.shape  : {x1.shape}')
    print(f'x2.shape  : {x2.shape}')
    print(f'outputs.shape : {outputs.shape}')
