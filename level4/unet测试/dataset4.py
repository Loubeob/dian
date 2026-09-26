import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class HandwritingDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=512):
        self.input_dir = os.path.join(root_dir, split, 'input')
        self.target_dir = os.path.join(root_dir, split, 'target')
        self.img_size = img_size
        self.split = split
        self.images = sorted([f for f in os.listdir(self.input_dir)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(self.images) == 0:
            raise RuntimeError(f'{split} 目录没有图片: {self.input_dir}')
        print(f'{split}: {len(self.images)} 张图片')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        input_img = Image.open(os.path.join(self.input_dir, img_name)).convert('RGB')
        target_img = Image.open(os.path.join(self.target_dir, img_name)).convert('RGB')

        input_img = input_img.resize((self.img_size, self.img_size), Image.BICUBIC)
        target_img = target_img.resize((self.img_size, self.img_size), Image.BICUBIC)

        if self.split == 'train':
            if random.random() > 0.5:
                input_img = TF.hflip(input_img)
                target_img = TF.hflip(target_img)
            if random.random() > 0.5:
                input_img = TF.vflip(input_img)
                target_img = TF.vflip(target_img)
            if random.random() > 0.7:
                input_img = TF.adjust_brightness(input_img, random.uniform(0.9, 1.1))
                input_img = TF.adjust_contrast(input_img, random.uniform(0.9, 1.1))

        input_tensor = transforms.ToTensor()(input_img)
        target_tensor = transforms.ToTensor()(target_img)

        return input_tensor, target_tensor, img_name
