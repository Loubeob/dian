# Level 1：MLP 手写数字识别（MNIST）

## 网络结构
MLP（多层感知机），输入 28×28 灰度图片展平为 784 维：
- 全连接层：784 → 256（ReLU + Dropout 0.2）
- 全连接层：256 → 128（ReLU + Dropout 0.2）
- 输出层：128 → 10（10个数字类别）

## 超参数
| 参数 | 值 |
|------|-----|
| Epochs | 10 |
| Batch Size | 64 |
| 学习率 | 0.001（Adam优化器） |
| Dropout | 0.2 |
| 损失函数 | CrossEntropyLoss |

## 数据预处理
- ToTensor：图片 0-255 → 0-1
- Normalize：均值 0.1307、标准差 0.3081 标准化

## 实验结果
- 测试集准确率：约 98%（验收要求 ≥90%，达标）
- 训练/测试 Loss 曲线见 training_curves.png

## 文件说明
- level1.1.py：训练脚本（含国内镜像下载MNIST）
- m.ipynb / mm.ipynb：训练（+推理）notebook版
- mlp_predict.py：单张图片推理脚本

## 运行方式
训练：python level1.1.py
推理：python mlp_predict.py 你的图片.png