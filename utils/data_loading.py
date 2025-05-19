import logging
import numpy as np
import torch
import os
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = os.path.join(mask_dir, [file for file in os.listdir(mask_dir) if file.startswith(idx)][0])
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.ids = set()
        self.image_groups = []
        for file in os.listdir(images_dir):
            if isfile(join(images_dir, file)) and not file.startswith('.'):
                prefix = file.split('-')[0]
                self.ids.add(prefix)
        self.ids = list(self.ids)

        for prefix in self.ids:
            group = []
            for file in sorted(os.listdir(images_dir)):
                if file.startswith(prefix+'-'):
                    group.append(file)
                    if len(group) == 3:
                        self.image_groups.append(group)
                        group = group[1:]  # Move window by 1
            # if len(group) == 2:  # Handle remaining two images
            #     self.image_groups.append(group)
        # self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.image_groups)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.image_groups)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        # name = self.image_groups[idx]  # 获取索引对应的样本ID
        # print(name)

        img_files = self.image_groups[idx]
        # print(img_files)
        mask_files = self.image_groups[idx]

        # 加载图像和掩码数据
        images = [Image.open(os.path.join(self.images_dir,file)) for file in img_files]
        # masks = [Image.open(os.path.join(self.mask_dir,file)) for file in mask_files]
 
        image = []
        for img in images:
            img_array = np.array(img)  # 将PIL图像转换为NumPy数组
            if (img_array > 1).any():
                img_array = img_array / 255.0  
            image.append(img_array)
        # 堆叠图像和掩码数据
        stacked_images = np.stack(image, axis=0)

        # stacked_masks = []
        # # 遍历每张掩码图像
        # for mask in masks:
        #     # 获取掩码图像的尺寸
        #     height, width = mask.size

        #     # 创建一个空白的数组，用于存储当前掩码图像的类别信息
        #     unique_mask = np.zeros((height, width), dtype=np.int64)

        #     # 遍历每个类别值，并将当前掩码图像的每个类别分配到相应的通道
        #     for i, v in enumerate(self.mask_values):
        #         unique_mask[mask == v] = i

        #     # 将处理后的掩码图像添加到 stacked_masks 列表中
        #     stacked_masks.append(unique_mask)
        # stacked_masks = np.stack(stacked_masks, axis=0)
        # merged_mask = np.zeros_like(masks[0], dtype=np.int32)
        # # 创建一个数组来统计每个位置的类别频次

        # # 将掩码堆叠成一个数组，形状为 (num_masks, height, width)
        # stacked_masks = np.stack(masks)

        # # 统计每个位置的类别频次
        # class_counts = np.apply_along_axis(lambda x: Counter(x).most_common(1)[0][0], axis=0, arr=stacked_masks)

        # # 将合并后的类别填充到合并后的掩码中
        # merged_mask = class_counts
        mask = Image.open(os.path.join(self.mask_dir, mask_files[1]))
        mask_array = np.array(mask)
        unique_mask = np.zeros(mask_array.shape, dtype=np.int64)
        for i, v in enumerate(self.mask_values):
            unique_mask[mask_array == v] = i

            
        return {
            'image': torch.as_tensor(stacked_images.copy()).float().contiguous(),
            'mask': torch.as_tensor(unique_mask.copy()).long().contiguous()
        }

       

class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
