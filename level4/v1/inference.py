import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import UNet


def get_args():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/home/lf/checkpoints/unet/unet_best.pth')
    parser.add_argument('--input-dir', type=str, default='/home/lf/level4/dataset/test/input')
    parser.add_argument('--output-dir', type=str, default='/home/lf/level4/unet_results')
    parser.add_argument('--img-size', type=int, default=256)
    return parser.parse_known_args()[0]


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channels=3, out_channels=3).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'模型加载成功, PSNR: {ckpt["psnr"]:.2f}dB')

    os.makedirs(args.output_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    images = sorted([f for f in os.listdir(args.input_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f'找到 {len(images)} 张测试图片')

    for fname in images:
        img = Image.open(os.path.join(args.input_dir, fname)).convert('RGB')
        original_size = img.size
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            output = torch.clamp(output, 0, 1)

        # 保存干净图
        result = transforms.ToPILImage()(output.squeeze(0).cpu())
        result = result.resize(original_size, Image.BICUBIC)
        result.save(os.path.join(args.output_dir, f'clean_{fname}'))

        # 可视化对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(img); ax1.set_title('Input (with handwriting)'); ax1.axis('off')
        ax2.imshow(result); ax2.set_title('Output (erased)'); ax2.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'compare_{fname}'), dpi=150)
        plt.close()
        print(f'处理完成: {fname}')

    print(f'\n所有结果已保存到: {args.output_dir}')


if __name__ == '__main__':
    main()
