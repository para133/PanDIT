import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# 计算 Spectral Angle Mapper (SAM)
def spectral_angle_mapper(img1, img2):
    img1 = img1.astype(np.float64) + 1e-8  # 防止除零
    img2 = img2.astype(np.float64) + 1e-8
    
    dot_product = np.sum(img1 * img2, axis=2)
    norm1 = np.linalg.norm(img1, axis=2)
    norm2 = np.linalg.norm(img2, axis=2)
    
    sam_map = np.arccos(np.clip(dot_product / (norm1 * norm2), -1, 1))  # 防止超出 [-1,1]
    return np.mean(sam_map)  # 取平均值作为整体SAM

# 计算 ERGAS (全局误差归一化)
def ergas(img1, img2, scale_factor=4):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse_per_band = np.mean((img1 - img2) ** 2, axis=(0, 1))  # 计算每个通道的MSE
    mean_ref = np.mean(img1, axis=(0, 1))  # 计算每个通道的均值
    ergas_value = 100 * np.sqrt(np.mean(mse_per_band / (mean_ref ** 2))) / scale_factor
    return ergas_value

# 计算 SSC (空间结构复杂度)
def spatial_structure_complexity(img1, img2):
    img1_gray = np.mean(img1, axis=2)  # 转换为灰度图
    img2_gray = np.mean(img2, axis=2)
    
    grad_x1 = np.gradient(img1_gray, axis=0)
    grad_y1 = np.gradient(img1_gray, axis=1)
    grad_x2 = np.gradient(img2_gray, axis=0)
    grad_y2 = np.gradient(img2_gray, axis=1)
    
    numerator = np.sum((grad_x1 - grad_x2) ** 2 + (grad_y1 - grad_y2) ** 2)
    denominator = np.sum(grad_x1 ** 2 + grad_y1 ** 2 + grad_x2 ** 2 + grad_y2 ** 2)
    
    ssc_value = 1 - numerator / (denominator + 1e-8)  # 防止除零
    return ssc_value

# 计算所有指标
def calculate_metrics(folder1, folder2):
    files = os.listdir(folder1)
    
    psnr_values = []
    ssim_values = []
    sam_values = []
    ergas_values = []
    ssc_values = []
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    for file in files:
        img1 = Image.open(os.path.join(folder1, file)).convert('RGB')
        img2 = Image.open(os.path.join(folder2, file)).convert('RGB')
        
        img1 = np.array(img1)
        img2 = np.array(img2)
        
        psnr_value = psnr(img1, img2)
        ssim_value = ssim(img1, img2, win_size=11, channel_axis=2)
        sam_value = spectral_angle_mapper(img1, img2)
        ergas_value = ergas(img1, img2)
        ssc_value = spatial_structure_complexity(img1, img2)
        
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        sam_values.append(sam_value)
        ergas_values.append(ergas_value)
        ssc_values.append(ssc_value)
        
        print(f'{file} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, SAM: {sam_value:.4f}, ERGAS: {ergas_value:.4f}, SSC: {ssc_value:.4f}')
    
    # 计算均值和标准差
    def mean_std(values):
        return np.mean(values), np.std(values)

    avg_psnr, std_psnr = mean_std(psnr_values)
    avg_ssim, std_ssim = mean_std(ssim_values)
    avg_sam, std_sam = mean_std(sam_values)
    avg_ergas, std_ergas = mean_std(ergas_values)
    avg_ssc, std_ssc = mean_std(ssc_values)

    # 打印最终结果
    print(f'Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}')
    print(f'Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}')
    print(f'Average SAM: {avg_sam:.4f} ± {std_sam:.4f}')
    print(f'Average ERGAS: {avg_ergas:.4f} ± {std_ergas:.4f}')
    print(f'Average SSC: {avg_ssc:.4f} ± {std_ssc:.4f}')

    
folder1 = 'output'
folder2 = r'E:\code\PanSharpening\PanDataset\WV2_data\test128\ms_HR'
calculate_metrics(folder1, folder2)