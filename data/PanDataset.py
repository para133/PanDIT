import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.multiprocessing
import pywt
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')
        
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pywt

# 获取路径
py_path = os.path.abspath(__file__) 
file_dir = os.path.dirname(py_path) 
class PanDataset(Dataset):
    def __init__(self, img_size, ms_folder, pan_folder, GT_folder=None, mode='train'):
        self.images = [f for f in os.listdir(ms_folder) if f.endswith('.tif')]
        self.ms_folder = ms_folder
        self.pan_folder = pan_folder
        self.GT_folder = GT_folder
        self.mode = mode
        self.img_size = img_size
        
        if mode == 'train':
            self.random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
            self.random_vertical_flip = transforms.RandomVerticalFlip(p=0.5)
            self.random_color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            self.random_crop = transforms.RandomResizedCrop(size=self.img_size, scale=(0.1, 0.4), ratio=(0.8, 1.2))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def random_transform(self, img, seed):
        torch.manual_seed(seed)
        img = self.random_color_transform(img) 
        img = self.random_horizontal_flip(img)
        img = self.random_vertical_flip(img)
        img = self.random_crop(img)
        return img
    
    def __getitem__(self, idx):
        # 加载图像
        ms_image_path = os.path.join(self.ms_folder, self.images[idx])
        pan_image_path = os.path.join(self.pan_folder, self.images[idx]) 
        GT_image_path = os.path.join(self.GT_folder, self.images[idx]) if self.GT_folder else None

        img_ms = Image.open(ms_image_path).convert('RGB')
        img_ms_up = img_ms.resize((img_ms.size[0] * 4, img_ms.size[1] * 4), Image.BICUBIC)
        img_pan = Image.open(pan_image_path)

        if GT_image_path:
            img_GT = Image.open(GT_image_path).convert('RGB')
        else:
            img_GT = img_ms_up

        # 训练模式下进行一致的数据增强
        if self.mode == 'train':
            seed = np.random.randint(2147483647) 

            img_ms_up = self.random_transform(img_ms_up, seed)
            img_ms_up.save('img_ms_up.tif')
            img_pan = self.random_transform(img_pan, seed)
            img_pan.save('img_pan.tif')
            img_GT = self.random_transform(img_GT, seed)
            img_GT.save('img_GT.tif')
            
        # 进行小波变换
        lms_main, (lms_h, lms_v, lms_d) = pywt.wavedec2(np.array(img_ms_up), "db1", level=1, axes=[0,1])
        pan_main, (pan_h, pan_v, pan_d) = pywt.wavedec2(np.array(img_pan), "db1", level=1, axes=[0,1])
        
        img_ms_up = self.to_tensor(img_ms_up)
        img_pan = self.to_tensor(img_pan)
        img_GT = self.to_tensor(img_GT)

        wavelets_dcp = torch.cat([self.to_tensor(lms_main), self.to_tensor(pan_h), 
                                  self.to_tensor(pan_d), self.to_tensor(pan_v)], dim=0)

        return {
            'img_ms_up': img_ms_up, 'img_pan': img_pan, 'wavelets': wavelets_dcp,
            'GT': img_GT,
            'file_name': self.images[idx]
        }
        
if __name__ == '__main__':
    ms_folder = r"E:\code\PanSharpening\PanDataset\GF2_data\train128\ms_LR"
    pan_folder = r"E:\code\PanSharpening\PanDataset\GF2_data\train128\pan"
    GT_folder = r"E:\code\PanSharpening\PanDataset\GF2_data\train128\ms_HR"
    img_size = 128
    dataset = PanDataset(img_size, ms_folder, pan_folder, GT_folder, mode='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    for data in dataloader:
        print(data['img_ms_up'].shape)
        print(data['img_pan'].shape)
        print(data['wavelets'].shape)
        print(data['GT'].shape)

