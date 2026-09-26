# Level 2：CNN 手写数字识别（与 MLP 对比）

## 网络结构
CNN（卷积神经网络）：
- 卷积层：Conv(1→32, 3×3) + ReLU + MaxPool(2×2)
- 卷积层：Conv(32→64, 3×3) + ReLU + MaxPool(2×2)
- 展平 → 全连接 → 10

## 超参数
| 参数 | 值 |
|------|-----|
| Epochs | 10 |
| Batch Size | 64 |
| 学习率 | 0.001（Adam） |
| 损失函数 | CrossEntropyLoss |

## 实验结果：CNN vs MLP 对比
| 对比项 | MLP | CNN |
|--------|-----|-----|
| 测试准确率 | ~98% | **99.31%** |
| 参数量 | 235,146 | 421,642 |
| 收敛速度 | 较慢 | 快（第1轮已达92%+） |
| 错误样本 | 较多 | 较少（见 cnn_error_samples.png） |

## 为什么 CNN 更适合图像任务
1. **局部感受野**：卷积核只关注相邻像素，天然利用图片"相邻像素相关"的特征
2. **权重共享**：同一卷积核扫全图，参数量更省、更不容易过拟合
3. **平移不变性**：MaxPool 下采样让特征对位置不敏感
4. 对比实验证明：同样10轮训练，CNN 准确率更高（99.31% vs 98%）

## 文件说明
- level2.ipynb：完整训练+对比实验notebook
- 输出图：cnn_training_curves.png、mlp_vs_cnn_comparison.png、cnn_error_samples.png

## 运行方式
用 Jupyter 打开 level2.ipynb 运行全部 cell