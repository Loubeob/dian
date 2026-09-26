import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """U-Net 基本单元: 两个 3x3 卷积 + BN + ReLU"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net: 编码器(下采样) + 瓶颈层 + 解码器(上采样) + 跳跃连接
    输入输出同尺寸, 适合图像擦除/分割任务
    """

    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器: 下采样, 通道数翻倍
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器: 上采样 + 跳跃连接
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # 输出层: 1x1 卷积
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 编码器
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # 解码器
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = UNet(in_channels=3, out_channels=3)
    print(f'U-Net 参数量: {count_parameters(m):,}')
    dummy = torch.randn(1, 3, 256, 256)
    out = m(dummy)
    print(f'输入 {dummy.shape} -> 输出 {out.shape}')
