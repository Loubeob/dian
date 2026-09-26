"""Level 3.2: U-Net 图像分割核心结构
编码器(下采样) + 解码器(上采样) + 跳跃连接
"""
import torch
import torch.nn as nn


# 一次"卷积+ReLU"操作
def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU(),
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # 编码器: 每层通道翻倍, 尺寸减半(下采样)
        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        # 瓶颈
        self.mid = conv_block(256, 512)
        # 解码器: 上采样 + 跳跃连接拼接 + 卷积
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # 编码(记录跳跃连接)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.mid(self.pool(e3))
        # 解码(拼接跳跃连接)
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)  # 随机图前向测试
    out = UNet()(x)
    print(f'输入: {x.shape} -> 输出: {out.shape} (和输入同尺寸, 逐像素分割)')