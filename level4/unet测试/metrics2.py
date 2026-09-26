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
