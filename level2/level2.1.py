"""CNN MNIST 最核心代码: 训练 + 测试准确率"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)   # 32个3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, 3)  # 64个3x3卷积核
        self.pool = nn.MaxPool2d(2)        # 2x2下采样
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(
    datasets.MNIST('/home/lf/data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('/home/lf/data', train=False, download=True, transform=transform),
    batch_size=64, shuffle=False)

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
    acc = correct / len(test_loader.dataset) * 100
    print(f'epoch {epoch+1}, loss={loss.item():.4f}, test acc={acc:.2f}%')