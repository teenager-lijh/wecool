import torch
import torch.nn as nn
from network.module import DoubleConv
from network.up import Up
from network.down import Down


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()

        self.inc = DoubleConv(in_channels=in_channels, out_channels=32)
        self.down1 = Down(in_channels=32, out_channels=64)
        self.down2 = Down(in_channels=64, out_channels=128)
        self.down3 = Down(in_channels=128, out_channels=256)
        self.down4 = Down(in_channels=256, out_channels=512)

        self.bridge = DoubleConv(in_channels=512, out_channels=512)

        self.up4 = Up(in_channels=512 * 2, out_channels=256)
        self.up3 = Up(in_channels=256 * 2, out_channels=128)
        self.up2 = Up(in_channels=128 * 2, out_channels=64)
        self.up1 = Up(in_channels=64 * 2, out_channels=32)

        self.outc = DoubleConv(in_channels=32, out_channels=out_channels)

    def forward(self, x):
        x = self.inc(x)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        bridge = self.bridge(d4)

        u4 = self.up4(bridge, d4)
        u3 = self.up3(u4, d3)
        u2 = self.up2(u3, d2)
        u1 = self.up1(u2, d1)

        out = self.outc(u1)

        return out


if __name__ == '__main__':
    inputs = torch.rand(size=(8, 3, 224, 224))
    net = UNet(in_channels=3, out_channels=2)
    outputs = net(inputs)

    print(f'inputs.shape: {inputs.shape}')
    print(f'outputs.shape: {outputs.shape}')

    # TODO: predict a sample

