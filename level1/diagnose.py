"""诊断: 查看图片预处理成28x28后的样子 + 各类别概率
用法: python diagnose.py 4.png
"""
import sys
import torch
from PIL import Image, ImageOps
import numpy as np
from torchvision import transforms
from mlp_predict import MLP

model = MLP()
ckpt = torch.load('./mlp_mnist_best.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

img = Image.open(sys.argv[1]).convert('L')
arr = np.array(img)
print(f'原图: {img.size}, 平均亮度: {arr.mean():.0f}')
if arr.mean() >= 128:
    img = ImageOps.invert(img)
    print('→ 白底黑字, 已反转成黑底白字')
else:
    print('→ 黑底白字, 保持不变')
img28 = img.resize((28, 28), Image.LANCZOS)
a = np.array(img28)
print('\n28x28 模型看到的图 (#=笔画, .=背景):')
for row in a:
    print(''.join('#' if v > 100 else '.' for v in row))
t = transforms.ToTensor()(img28).unsqueeze(0)
t = (t - 0.1307) / 0.3081
with torch.no_grad():
    out = model(t)
    probs = torch.softmax(out, 1).squeeze(0)
print('\n各类别概率:')
for i, p in enumerate(probs):
    print(f'  {i}: {p:.2f} ' + '#' * int(p * 40))
print(f'\n预测结果: {out.argmax(1).item()}')