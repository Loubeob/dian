import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import UNet, count_parameters
from dataset import HandwritingDataset
from metrics import calculate_metrics


def get_args():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--data-dir', type=str, default='/home/lf/level4/dataset')
    parser.add_argument('--save-dir', type=str, default='/home/lf/checkpoints/unet')
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_known_args()[0]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0
    criterion = nn.L1Loss()
    for inputs, targets, _ in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        for i in range(outputs.size(0)):
            pred = torch.clamp(outputs[i], 0, 1)
            p, s = calculate_metrics(pred, targets[i])
            total_psnr += p
            total_ssim += s
    n = len(loader.dataset)
    return total_loss / n, total_psnr / n, total_ssim / n


def plot_curves(train_losses, val_losses, val_psnrs, val_ssims, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss')
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

    train_dataset = HandwritingDataset(args.data_dir, 'train', args.img_size)
    val_dataset = HandwritingDataset(args.data_dir, 'val', args.img_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = UNet(in_channels=3, out_channels=3).to(device)
    print(f'U-Net 参数量: {count_parameters(model):,}')

    # L1 Loss 对边缘更友好, 适合图像恢复任务
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     factor=0.5, patience=10)

    os.makedirs(args.save_dir, exist_ok=True)
    train_losses, val_losses, val_psnrs, val_ssims = [], [], [], []
    best_psnr = 0.0

    print(f'\n开始训练 {args.epochs} 轮...')
    print('-' * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_psnr, val_ssim = evaluate(model, val_loader, device)
        scheduler.step(val_psnr)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        val_ssims.append(val_ssim)

        print(f'Epoch {epoch:3d}/{args.epochs} | Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | PSNR: {val_psnr:.2f}dB | SSIM: {val_ssim:.4f}')

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

    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict(),
                'psnr': val_psnr, 'ssim': val_ssim},
               os.path.join(args.save_dir, 'unet_last.pth'))

    plot_curves(train_losses, val_losses, val_psnrs, val_ssims,
                os.path.join(args.save_dir, 'training_curves.png'))

    with open(os.path.join(args.save_dir, 'training_log.txt'), 'w') as f:
        f.write('Epoch,Train_Loss,Val_Loss,PSNR,SSIM\n')
        for i in range(len(train_losses)):
            f.write(f'{i + 1},{train_losses[i]:.4f},{val_losses[i]:.4f},'
                    f'{val_psnrs[i]:.2f},{val_ssims[i]:.4f}\n')


if __name__ == '__main__':
    main()
