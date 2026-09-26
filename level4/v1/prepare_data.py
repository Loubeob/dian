"""
数据划分脚本
将 deli/日期/dataset/{input,output} 结构转换为:
dataset/
├── train/input, train/target
├── val/input, val/target
└── test/input, test/target
"""
import os
import random
import shutil

SRC = '/home/lf/level4/dataset/deli'
DST = '/home/lf/level4/dataset'

def collect_pairs(src):
    """收集所有 (input路径, output路径) 配对"""
    pairs = []
    for date_dir in sorted(os.listdir(src)):
        ds_dir = os.path.join(src, date_dir, 'dataset')
        if not os.path.isdir(ds_dir):
            continue
        input_dir = os.path.join(ds_dir, 'input')
        output_dir = os.path.join(ds_dir, 'output')
        if not os.path.isdir(input_dir) or not os.path.isdir(output_dir):
            print(f'跳过 {ds_dir}: 缺少 input/output')
            continue
        for fname in sorted(os.listdir(input_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                in_path = os.path.join(input_dir, fname)
                out_path = os.path.join(output_dir, fname)
                if os.path.exists(out_path):
                    pairs.append((in_path, out_path, fname))
    return pairs

def main():
    random.seed(42)
    pairs = collect_pairs(SRC)
    print(f'找到 {len(pairs)} 对图片')
    if len(pairs) == 0:
        print('没有找到数据，请检查数据集路径')
        return

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    splits = {
        'train': pairs[:n_train],
        'val': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:],
    }

    for split, sp_pairs in splits.items():
        in_dir = os.path.join(DST, split, 'input')
        out_dir = os.path.join(DST, split, 'target')
        os.makedirs(in_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        for in_path, out_path, fname in sp_pairs:
            shutil.copy2(in_path, os.path.join(in_dir, fname))
            shutil.copy2(out_path, os.path.join(out_dir, fname))
        print(f'{split}: {len(sp_pairs)} 张')

    print('数据划分完成！')

if __name__ == '__main__':
    main()
