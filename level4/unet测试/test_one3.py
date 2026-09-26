import os
import sys
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model3 import UNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512


def load_model(ckpt_path='./checkpoints/unet_v4/unet_best.pth'):
    model = UNet(in_channels=3).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'模型加载成功, PSNR: {ckpt["psnr"]:.2f}dB')
    return model


def erase(model, image_path, out_dir='./unet_results_v4'):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    img = ImageEnhance.Sharpness(img).enhance(1.3)

    tensor = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 残差学习：模型输出掩码，输出=输入*(1-掩码)
        mask = model(tensor)
        output = torch.clamp(tensor * (1 - mask), 0, 1)

    result = transforms.ToPILImage()(output.squeeze(0).cpu())
    result = result.resize(original_size, Image.BICUBIC)

    base = os.path.splitext(os.path.basename(image_path))[0]
    result_path = os.path.join(out_dir, f'clean_{base}.png')
    result.save(result_path)

    # 同时保存掩码可视化
    mask_img = transforms.ToPILImage()(mask.squeeze(0).cpu())
    mask_path = os.path.join(out_dir, f'mask_{base}.png')
    mask_img.save(mask_path)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes[0].imshow(img); axes[0].set_title('Input (with handwriting)'); axes[0].axis('off')
    axes[1].imshow(mask_img, cmap='gray'); axes[1].set_title('Mask (1=erase)'); axes[1].axis('off')
    axes[2].imshow(result); axes[2].set_title('Output (erased)'); axes[2].axis('off')
    plt.tight_layout()
    cmp_path = os.path.join(out_dir, f'compare_{base}.png')
    plt.savefig(cmp_path, dpi=150)
    plt.close()

    print(f'擦除完成!')
    print(f'  原图: {image_path}')
    print(f'  擦除结果: {result_path}')
    print(f'  掩码图: {mask_path}')
    print(f'  对比图: {cmp_path}')


if __name__ == '__main__':
    image = None
    if len(sys.argv) > 1:
        image = sys.argv[sys.argv.index('--image') + 1] if '--image' in sys.argv else sys.argv[1]
    if not image:
        print('用法: python test_one3.py --image 你的图片.jpg')
        sys.exit(1)
    model = load_model()
    erase(model, image)
