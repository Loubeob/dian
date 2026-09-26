import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model3 import UNet, count_parameters
from dataset3 import HandwritingDataset
from metrics3 import calculate_metrics, SSIMLoss


def get_args():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--l1-weight', type=float, default=1.0)
    parser.add_argument('--ssim-weight', type=float, default=0.1)
    parser.add_argument('--mask-weight', type=float, default=0.01)
    parser.add_argument('--data-dir', type=str, default='./dataset')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/unet_v4')
    parser.add_argument('--save-interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_known_args()[0]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        mask = model(inputs)
        outputs = torch.clamp(inputs * (1 - mask), 0, 1)
        for i in range(outputs.size(0)):
            p, s = calculate_metrics(outputs[i], targets[i])
            total_psnr += p
            total_ssim += s
    n = len(loader.dataset)
    return total_psnr / n, total_ssim / n


def plot_curves(train_losses, val_psnrs, val_ssims, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, val_psnrs, 'g-')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('PSNR (dB)'); axes[1].set_title('PSNR')
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(epochs, val_ssims, 'm-')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('SSIM'); axes[2].set_title('SSIM')
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'曲线已保存: {save_path}')


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

    train_dataset = HandwritingDataset(args.data_dir, 'train', args.img_size)
    val_dataset = HandwritingDataset(args.data_dir, 'val', args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = UNet(in_channels=3).to(device)
    print(f'U-Net 参数量: {count_parameters(model):,}')
    print('残差学习模式: 输出=输入*(1-掩码), 印刷字直接保留')

    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    os.makedirs(args.save_dir, exist_ok=True)
    train_losses, val_psnrs, val_ssims = [], [], []
    best_psnr = 0.0

    print(f'\n开始训练 {args.epochs} 轮 (512x512, 残差学习, L1+0.1*SSIM, CosineLR)...')
    print('-' * 80)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            # 残差学习：模型只输出掩码，输出=输入*(1-掩码)
            mask = model(inputs)
            outputs = inputs * (1 - mask)
            # 主loss：输出和目标的L1+SSIM
            loss = args.l1_weight * l1_loss(outputs, targets) \
                   + args.ssim_weight * ssim_loss(outputs, targets)
            # 辅助loss：掩码稀疏性，鼓励掩码尽量小（只擦手写，不碰印刷字）
            loss = loss + args.mask_weight * mask.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        train_loss = total_loss / len(train_dataset)
        scheduler.step()

        val_psnr, val_ssim = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)

        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:3d}/{args.epochs} | Loss: {train_loss:.4f} | '
              f'PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f} | LR: {current_lr:.6f}')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'psnr': val_psnr, 'ssim': val_ssim},
                       os.path.join(args.save_dir, 'unet_best.pth'))

        if epoch % args.save_interval == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'psnr': val_psnr, 'ssim': val_ssim},
                       os.path.join(args.save_dir, f'unet_epoch{epoch}.pth'))

    print('-' * 80)
    print(f'训练完成! 最佳 PSNR: {best_psnr:.2f}dB')

    plot_curves(train_losses, val_psnrs, val_ssims,
                os.path.join(args.save_dir, 'training_curves.png'))

    with open(os.path.join(args.save_dir, 'training_log.txt'), 'w') as f:
        f.write('Epoch,Train_Loss,PSNR,SSIM\n')
        for i in range(len(train_losses)):
            f.write(f'{i + 1},{train_losses[i]:.4f},{val_psnrs[i]:.2f},{val_ssims[i]:.4f}\n')


if __name__ == '__main__':
    main()
