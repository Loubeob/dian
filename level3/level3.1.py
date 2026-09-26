"""Level 3.1: 经典分类网络 AlexNet / VGG / ResNet
三个网络的核心结构 + 前向测试(随机数据, 证明结构正确)
"""
import torch
import torch.nn as nn


# ===== AlexNet: 深层CNN + ReLU + Dropout =====
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3), nn.ReLU(),
            nn.Conv2d(384, 384, 3), nn.ReLU(),
            nn.Conv2d(384, 256, 3), nn.ReLU(), nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ===== VGG: 重复堆叠小卷积核(3x3), 网络模块化 =====
class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # 每段 = 2个3x3卷积 + 池化
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


# ===== ResNet: 残差跳跃连接 =====
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        return torch.relu(out + x)  # 残差: 输出+输入


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3)
        self.block1 = ResidualBlock(32)
        self.block2 = ResidualBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))


# ===== 前向测试: 随机数据跑通每个网络 =====
if __name__ == '__main__':
    x = torch.randn(2, 3, 227, 227)  # AlexNet需要224+输入
    print('AlexNet:', AlexNet()(x).shape)
    x = torch.randn(2, 3, 64, 64)
    print('VGG:', VGG()(x).shape)
    print('ResNet:', ResNet()(x).shape)
    print('三个网络前向传播全部通过')