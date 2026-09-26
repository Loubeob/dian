"""
Level 3: 经典网络 AlexNet、VGG、ResNet、U-Net
理解网络结构和核心知识点，为 Level 4 U-Net 项目做准备
"""
import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== 1. AlexNet ====================
print('\n' + '='*60)
print('1. AlexNet (2012, ImageNet冠军, 深度学习里程碑)')
print('='*60)

class AlexNet(nn.Module):
    """
    AlexNet 核心创新:
    1. 首次用ReLU激活函数 (之前用sigmoid/tanh, 梯度消失严重)
    2. Dropout 防止过拟合
    3. 数据增强 (随机裁剪、翻转、颜色抖动)
    4. GPU并行训练 (当时用两块GTX 580)
    5. Local Response Normalization (LRN, 现在基本不用了)
    """
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # 224→55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # 55→27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),            # 27→27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # 27→13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),           # 13→13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),           # 13→13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),           # 13→13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                   # 13→6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

alexnet = AlexNet(num_classes=10)
print(f'参数量: {count_parameters(alexnet):,}')
print('输入: [batch, 3, 224, 224] → 输出: [batch, num_classes]')
print('特点: 8层网络(5卷积+3全连接), 首次证明深度CNN在图像任务上的巨大优势')


# ==================== 2. VGG ====================
print('\n' + '='*60)
print('2. VGG (2014, 用小卷积核堆叠代替大卷积核)')
print('='*60)

class VGG(nn.Module):
    """
    VGG 核心思想:
    1. 全部用 3x3 小卷积核, 堆叠多个3x3达到大卷积核的感受野
       2个3x3 = 1个5x5感受野, 3个3x3 = 1个7x7感受野
    2. 小卷积核参数量更少, 且可以加入更多ReLU非线性
    3. 网络模块化: 每阶段卷积层数递增, 通道数翻倍
    4. VGG16/VGG19: 16层/19层权重层
    """
    def __init__(self, num_classes=1000):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 阶段1: 2个3x3卷积, 64通道, 224→112
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 阶段2: 2个3x3卷积, 128通道, 112→56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 阶段3: 3个3x3卷积, 256通道, 56→28
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 阶段4: 3个3x3卷积, 512通道, 28→14
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 阶段5: 3个3x3卷积, 512通道, 14→7
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

vgg = VGG(num_classes=10)
print(f'参数量: {count_parameters(vgg):,}')
print('输入: [batch, 3, 224, 224] → 输出: [batch, num_classes]')
print('特点: 13个卷积层+3个全连接层=VGG16, 结构规整, 特征提取能力强')


# ==================== 3. ResNet ====================
print('\n' + '='*60)
print('3. ResNet (2015, 残差连接, 解决深度网络退化问题)')
print('='*60)

class BasicBlock(nn.Module):
    """残差基本块: F(x) + x, 梯度可以直接通过x回传"""
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        # 如果维度不匹配, 用1x1卷积调整
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 残差连接: 输出 = F(x) + x
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet 核心思想:
    1. 退化问题: 网络越深, 训练误差反而上升 (不是过拟合, 是优化困难)
    2. 残差连接: H(x) = F(x) + x, 让网络学习残差F(x)=H(x)-x
    3. 如果某层不需要, F(x)可以学0, 相当于恒等映射, 不会退化
    4. 梯度可以通过shortcut直接回传, 缓解梯度消失
    5. 可以训练上百层甚至上千层的网络
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ResNet18: [2,2,2,2] 每个阶段2个残差块
resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
print(f'ResNet18 参数量: {count_parameters(resnet):,}')
print('输入: [batch, 3, 32, 32] (适配CIFAR) 或 224x224 → 输出: [batch, num_classes]')
print('特点: 残差连接让超深网络可训练, 是现代CNN的基础结构')


# ==================== 4. U-Net (重点, Level 4要用) ====================
print('\n' + '='*60)
print('4. U-Net (2015, 医学图像分割, 编码器-解码器+跳跃连接)')
print('='*60)

class DoubleConv(nn.Module):
    """U-Net基本单元: 两个3x3卷积+BN+ReLU"""
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
    U-Net 核心结构:
    1. 编码器(下采样): 卷积提取特征 + MaxPool降低分辨率, 通道数翻倍
    2. 瓶颈层: 最底层, 通道数最大
    3. 解码器(上采样): 转置卷积恢复分辨率 + 跳跃连接拼接编码器特征
    4. 跳跃连接(Skip Connection): 把编码器的特征图拼接到解码器
       让解码器同时利用高层语义和低层细节, 分割边缘更精确
    5. 最后1x1卷积: 通道数降到类别数, 输出分割图
    
    为什么叫U-Net: 结构像字母U, 左边编码器下采样, 右边解码器上采样
    """
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器: 下采样
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器: 上采样
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature))

        # 输出层: 1x1卷积
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # 编码器: 保存跳跃连接
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # 瓶颈层
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # 反转

        # 解码器: 上采样 + 拼接跳跃连接
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # 转置卷积上采样
            skip = skip_connections[idx // 2]
            # 如果尺寸不匹配, 裁剪
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)  # 通道维度拼接
            x = self.ups[idx + 1](x)  # 双卷积

        return self.final_conv(x)

unet = UNet(in_channels=1, out_channels=1)
print(f'U-Net 参数量: {count_parameters(unet):,}')
print('输入: [batch, 1, 256, 256] → 输出: [batch, 1, 256, 256] (和输入同尺寸的分割图)')

# 测试U-Net前向传播
dummy = torch.randn(1, 1, 256, 256)
out = unet(dummy)
print(f'测试: 输入 {dummy.shape} → 输出 {out.shape}')
print('特点: 编码器-解码器结构+跳跃连接, 输入输出同尺寸, 适合图像分割/擦除任务')


# ==================== 四个网络参数量对比 ====================
print('\n' + '='*60)
print('四个网络参数量对比')
print('='*60)
print(f'AlexNet:   {count_parameters(alexnet):>12,}')
print(f'VGG16:     {count_parameters(vgg):>12,}')
print(f'ResNet18:  {count_parameters(resnet):>12,}')
print(f'U-Net:     {count_parameters(unet):>12,}')

print('\n' + '='*60)
print('Level 3 完成! 四个经典网络结构已定义并验证')
print('U-Net 是 Level 4 手写擦除项目的核心模型')
print('='*60)
