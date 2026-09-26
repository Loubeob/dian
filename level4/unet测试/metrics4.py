import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_metrics(pred, target):
    pred = pred.detach()
    target = target.detach()
    mse_tensor = F.mse_loss(pred, target)
    mse = mse_tensor.item()
    if mse > 1e-8:
        psnr = 10 * torch.log10(1.0 / mse_tensor).item()
    else:
        psnr = 100.0
    window_size = 11
    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    ssim_val = ssim_map.mean().item()
    return psnr, ssim_val


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.c1 = 0.01 ** 2
        self.c2 = 0.03 ** 2

    def forward(self, img1, img2):
        mu1 = F.avg_pool2d(img1, self.window_size, stride=1, padding=self.window_size // 2)
        mu2 = F.avg_pool2d(img2, self.window_size, stride=1, padding=self.window_size // 2)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, stride=1, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, stride=1, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, stride=1, padding=self.window_size // 2) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))
        return 1 - ssim_map.mean()


class VGGPerceptualLoss(nn.Module):
    """VGG感知损失：用预训练VGG19的中间层特征对比，保留边缘和锐利度，解决L1模糊问题"""
    def __init__(self, layer_index=21, device='cpu'):
        super(VGGPerceptualLoss, self).__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:layer_index].to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        # VGG的归一化均值和标准差
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)

    def forward(self, pred, target):
        # 归一化到VGG的输入范围
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std
        feat_pred = self.vgg(pred_norm)
        feat_target = self.vgg(target_norm)
        return F.l1_loss(feat_pred, feat_target)
