import torch
import torch.nn as nn
from bluenet2.module import DoubleConv
from bluenet2.module import Attention


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2),
        #     DoubleConv(in_channels, out_channels)
        # )

        # 使用卷积进行下采样 ==> 而不是使用原来的池化层
        self.down = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            kernel_size=2
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        # 这一层没动，还是用了它
        self.conv = DoubleConv(
            in_channels=out_channels,
            out_channels=out_channels
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        # 下采样完成后使用一次 Attention ==> 全局尺度特征融合
        self.attn = Attention(
            dim=out_channels,  # channel 就相当于对每个像素点的编码长度
            num_heads=4,
            qkv_bias=False,
            qk_scale=None,
            sr_ratio=reduction
        )
        self.norm3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # return self.maxpool_conv(x)
        x = self.down(x)
        x = self.norm1(x)
        x = self.conv(x)
        x = self.norm2(x)
        r = x
        x = self.attn(x)
        x = x + r  # 做残差
        x = self.norm3(x)

        return x


if __name__ == '__main__':
    inputs = torch.rand(size=(8, 16, 224, 224))
    down = Down(in_channels=16, out_channels=32, reduction=64)
    outputs = down(inputs)
    print(f'inputs.shape  : {inputs.shape}')
    print(f'outputs.shape : {outputs.shape}')
