import os
import sys
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model2 import UNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 512


def load_model(ckpt_path='./checkpoints/unet_v2/unet_best.pth'):
    model = UNet(in_channels=3, out_channels=3).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'模型加载成功, PSNR: {ckpt["psnr"]:.2f}dB')
    return model


def erase(model, image_path, out_dir='./unet_results_v2'):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(image_path).convert('RGB')
    original_size = img.size

    img = ImageEnhance.Sharpness(img).enhance(1.3)

    tensor = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = torch.clamp(model(tensor), 0, 1)

    result = transforms.ToPILImage()(output.squeeze(0).cpu())
    result = result.resize(original_size, Image.BICUBIC)

    base = os.path.splitext(os.path.basename(image_path))[0]
    result_path = os.path.join(out_dir, f'clean_{base}.png')
    result.save(result_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(img); ax1.set_title('Input (with handwriting)'); ax1.axis('off')
    ax2.imshow(result); ax2.set_title('Output (erased)'); ax2.axis('off')
    plt.tight_layout()
    cmp_path = os.path.join(out_dir, f'compare_{base}.png')
    plt.savefig(cmp_path, dpi=150)
    plt.close()

    print(f'擦除完成!')
    print(f'  原图: {image_path}')
    print(f'  擦除结果: {result_path}')
    print(f'  对比图: {cmp_path}')


if __name__ == '__main__':
    image = None
    if len(sys.argv) > 1:
        image = sys.argv[sys.argv.index('--image') + 1] if '--image' in sys.argv else sys.argv[1]
    if not image:
        print('用法: python test_one2.py --image 你的图片.jpg')
        sys.exit(1)
    model = load_model()
    erase(model, image)
