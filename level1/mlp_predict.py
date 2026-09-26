"""MLP MNIST 单张图片推理
用法: python mlp_predict.py 你的图片.png
"""
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms


class MLP(nn.Module):
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10, dropout=0.2):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


def preprocess(img):
    """把任意手写图转成 MNIST 风格: 黑底白字 + 数字居中 28x28"""
    img = img.convert('L')
    # 统一成"黑底白字"(MNIST风格): 平均亮说明是白底黑字, 反转
    if np.mean(np.array(img)) >= 128:
        img = ImageOps.invert(img)
    a = np.array(img)
    ys, xs = np.where(a > 128)  # 笔画=亮像素
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = 2
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(a.shape[1], x2 + pad); y2 = min(a.shape[0], y2 + pad)
    crop = img.crop((x1, y1, x2, y2)).resize((20, 20))
    out = Image.new('L', (28, 28), 0)  # 黑底
    out.paste(crop, (4, 4))  # 居中
    return np.array(out, dtype=np.float32) / 255.0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python mlp_predict.py 你的图片.png')
        sys.exit(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP().to(device)
    import os
    ckpt_path = './mlp_mnist_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = '/home/lf/checkpoints/mlp_mnist_best.pth'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    img = Image.open(sys.argv[1])
    x = preprocess(img)
    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)
    tensor = (tensor - 0.1307) / 0.3081
    with torch.no_grad():
        out = model(tensor)
    pred = out.argmax(dim=1).item()
    print(f'预测结果: {pred}')