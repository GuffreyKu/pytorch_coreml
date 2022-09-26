import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channel, k_size, pad=1, s=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channel,
                              kernel_size=k_size, padding=pad, stride=s, dilation=dilation)
        self.batchNorm = nn.BatchNorm2d(out_channel)
        self.actfunction = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.actfunction(x)
        return x


class Fullyconnect(nn.Module):
    def __init__(self, in_channels, out_channel):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channel)
        self.actfunction = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.actfunction(x)
        return x
