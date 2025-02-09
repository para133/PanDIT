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

from utils.metric import AnalysisPanAcc

# 计算所有指标
def calculate_metrics(folder1, folder2):
    files = os.listdir(folder1)
    
    psnr_values = []
    ssim_values = []
    sam_values = []
    ergas_values = []
    cc_values = []
    
    analysis_d = AnalysisPanAcc()    
    
    for file in files:
        img1 = Image.open(os.path.join(folder1, file)).convert('RGB')
        img2 = Image.open(os.path.join(folder2, file)).convert('RGB')
        transform = transforms.ToTensor()
        img1 = transform(img1)
        img2 = transform(img2)
        
        acc_d = analysis_d.sam_ergas_psnr_cc_one_image(img1, img2)
        psnr_value = acc_d["PSNR"]
        ssim_value = analysis_d.ssim(img1, img2)
        sam_value = acc_d["SAM"]
        ergas_value = acc_d["ERGAS"]
        cc_value = acc_d["CC"]
        
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        sam_values.append(sam_value)
        ergas_values.append(ergas_value)
        cc_values.append(cc_value)
        
        print(f'{file} - PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}, SAM: {sam_value:.4f}, ERGAS: {ergas_value:.4f}, CC: {cc_value:.4f}')
    
    # 计算均值和标准差
    def mean_std(values):
        return np.mean(values), np.std(values)

    avg_psnr, std_psnr = mean_std(psnr_values)
    avg_ssim, std_ssim = mean_std(ssim_values)
    avg_sam, std_sam = mean_std(sam_values)
    avg_ergas, std_ergas = mean_std(ergas_values)
    avg_ssc, std_ssc = mean_std(cc_values)

    # 打印最终结果
    print(f'Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f}')
    print(f'Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}')
    print(f'Average SAM: {avg_sam:.4f} ± {std_sam:.4f}')
    print(f'Average ERGAS: {avg_ergas:.4f} ± {std_ergas:.4f}')
    print(f'Average CC: {avg_ssc:.4f} ± {std_ssc:.4f}')

folder1 = r'E:\code\PanSharpening\PanDataset\WV2_data\test128\ms_HR'
folder2 = 'output'
calculate_metrics(folder1, folder2)