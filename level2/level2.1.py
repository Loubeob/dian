"""Level 2.1: CNN MNIST 最简训练 (只保证能训练、效果达标)"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 6 * 6, 10)
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

model = CNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('/home/lf/data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('/home/lf/data', train=False, download=True, transform=transform), batch_size=64)

for epoch in range(5):
    for x, y in train_loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    correct = sum((model(x).argmax(1) == y).sum().item() for x, y in test_loader)
    print(f'epoch {epoch+1}, loss={loss.item():.4f}, acc={correct/10000*100:.2f}%')