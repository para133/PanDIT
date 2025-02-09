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

# 获取路径
py_path = os.path.abspath(__file__) 
file_dir = os.path.dirname(py_path) 
        
class PanDataset(Dataset):
    def __init__(self, ms_folder, pan_folder , GT_folder = None,
                 transform=transforms.Compose([
                    transforms.ToTensor(),
                ])):
        self.images = [f for f in os.listdir(ms_folder) if f.endswith('.tif')]
        self.ms_folder = ms_folder
        self.pan_folder = pan_folder
        self.GT_folder = GT_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 高分辨率图像与低分辨率图像同名
        if self.GT_folder is not None:
            ms_image_path = os.path.join(self.ms_folder, self.images[idx])
            pan_image_path = os.path.join(self.pan_folder, self.images[idx]) 
            GT_image_path = os.path.join(self.GT_folder, self.images[idx])
            img_ms = Image.open(ms_image_path).convert('RGB')
            img_ms_up = img_ms.resize((img_ms.size[0] * 4, img_ms.size[1] * 4), Image.BICUBIC)
            img_pan = Image.open(pan_image_path)  
            img_GT = Image.open(GT_image_path).convert('RGB')
            file_name = self.images[idx]
            
        else:
            ms_image_path = os.path.join(self.ms_folder, self.images[idx])
            pan_image_path = os.path.join(self.pan_folder, self.images[idx]) 
            img_ms = Image.open(ms_image_path).convert('RGB')
            img_ms_up = img_ms.resize((img_ms.size[0] * 4, img_ms.size[1] * 4), Image.BICUBIC)
            img_pan = Image.open(pan_image_path)
            img_GT = img_ms_up
            file_name = self.images[idx]
        
        lms_main, (lms_h, lms_v, lms_d) = pywt.wavedec2(
            np.array(img_ms_up), "db1", level=1, axes=[0,1]
        )
        pan_main, (pan_h, pan_v, pan_d) = pywt.wavedec2(
            np.array(img_pan), "db1", level=1, axes=[0,1]
        )
        if self.transform:
            wavelets_dcp = torch.cat(
                [*map(self.transform, [lms_main, pan_h, pan_d, pan_v])], dim=0
            )
            img_ms = self.transform(img_ms)
            img_ms_up = self.transform(img_ms_up)
            img_pan = self.transform(img_pan)
            img_GT = self.transform(img_GT)

            
        return {
            'img_ms': img_ms, 'img_ms_up': img_ms_up, 'img_pan': img_pan, 'wavelets': wavelets_dcp,
            'GT': img_GT,
            'file_name': file_name
        }