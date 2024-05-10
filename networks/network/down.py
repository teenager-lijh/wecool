import torch
import torch.nn as nn
from network.module import DoubleConv


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


if __name__ == '__main__':
    inputs = torch.rand(size=(8, 16, 224, 224))
    down = Down(in_channels=16, out_channels=32)
    outputs = down(inputs)
    print(f'inputs.shape  : {inputs.shape}')
    print(f'outputs.shape : {outputs.shape}')