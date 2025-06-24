import os

import torch
from torch.utils.data import DataLoader
import einops
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse

from diffusion.diffusion_ddpm_pan import make_beta_schedule
from models.PanVIT import PanDiT
from diffusion.diffusion_ddpm_pan import GaussianDiffusion
from utils.misc import model_load
from torchvision.transforms import ToPILImage, ToTensor
from data.PanDataset import PanDataset
from utils.metrics import AnalysisMetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

@torch.no_grad()
def predict(
    dataset_folder,
    save_dir,
    weight_path,
    image_size,
    patch_size,
    args,
    ms_num_channel = 3,
    pan_num_channel = 1,
    schedule_type="cosine",
    batch_size=1,
    n_steps=500,
):
    denoise_fn = PanDiT(
        in_channel=ms_num_channel,
        out_channel=ms_num_channel,
        image_size=image_size,
        patch_size=patch_size,
        inner_channel=args.inner_channel,
        noise_level_channel=args.noise_level_channel,
        lms_channel=ms_num_channel,
        pan_channel=pan_num_channel,
        time_hidden_ratio=args.time_hidden_ratio,
        num_dit_layers=args.num_dit_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        with_noise_level_emb=True,
        self_condition=args.self_condition,
    ).to(device)

    denoise_fn = model_load(weight_path, denoise_fn, device=device)
    denoise_fn.eval()
    print(f"load weight {weight_path}")
    
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=ms_num_channel,
        # pred_mode="x_start",
        pred_mode="noise",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
        self_condition=args.self_condition,
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule=schedule_type, n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)

    dataset = PanDataset(
        img_size=image_size,
        ms_folder=os.path.join(dataset_folder, 'ms'),
        pan_folder=os.path.join(dataset_folder, 'pan'),
        ms_is_GT=args.ms_is_GT,
        mode='test',
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    analysis_d = AnalysisMetrics()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            torch.cuda.empty_cache()
            pan, lms, wavelets, hr = map(lambda x: x.cuda(), (batch['img_pan'], batch['img_ms_up'], batch['wavelets'], batch['GT']))
            cond, _ = einops.pack(
                [
                    lms,
                    pan,
                    F.interpolate(wavelets, size=lms.shape[-1], mode="bicubic"),
                ],
                "b * h w",
            )
            cond = cond.to(torch.float32)
            # sr = diffusion(cond, mode="ddim_sample", section_counts="ddim2")
            sr = diffusion(cond, mode="ddpm_sample")
            sr = lms + sr
            sr = sr.clip(0, 1)
            hr = hr.to(sr.device)
            analysis_d.batch_metrics(sr, hr, mode=args.mode)
            for j in range(sr.shape[0]):
                sr_img = ToPILImage(mode=args.mode)(sr[j].cpu())
                sr_img.save(os.path.join(save_dir, f"{batch['file_name'][j]}"))
    print(analysis_d.get_metrics_str())
            
def parse_args():
    parser = argparse.ArgumentParser(description="predict script for PanDiT")
    parser.add_argument('--dataset_folder', type=str, required=True, help='Path to the validation dataset folder')
    parser.add_argument('--ms_num_channel', type=int, default=4, help='Number of multispectral channels')
    parser.add_argument('--pan_num_channel', type=int, default=1, help='Number of panchromatic channels')
    parser.add_argument('--image_size', type=int, default=128, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for processing')
    parser.add_argument('--schedule_type', type=str, default='cosine', choices=['cosine', 'linear'], help='Schedule type for beta schedule')
    parser.add_argument('--n_steps', type=int, default=100, help='Number of diffusion steps')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--weight_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--inner_channel', type=int, default=16, help='Number of inner channels')
    parser.add_argument('--noise_level_channel', type=int, default=128, help='Number of noise level channels')
    parser.add_argument('--time_hidden_ratio', type=int, default=4, help='Time hidden ratio')
    parser.add_argument('--num_dit_layers', type=int, default=12, help='Number of DIT layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio in DIT Block')
    parser.add_argument('--with_noise_level_emb', type=bool, default=True, help='Whether to include noise level embedding')
    parser.add_argument('--self_condition', type=bool, default=True, help='Whether to self condition')
    parser.add_argument('--save_dir', type=str, default='WV2_ddim2', help='Path to save the output images')
    parser.add_argument('--ms_is_GT', type=bool, default=True, help='Whether multispectral image is GT')
    parser.add_argument('--mode', type=str, default="CMYK", help='The mode of the image')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    py_path = os.path.abspath(__file__) 
    file_dir = os.path.dirname(py_path)
    dataset_folder = args.dataset_folder
    args.mode = "CMYK"
    args.save_dir = "WV2_ddpm"
    # args.ms_is_GT = False
    os.makedirs(args.save_dir, exist_ok=True)
    predict(
        dataset_folder=dataset_folder,
        save_dir=args.save_dir,
        weight_path=args.weight_path,   
        image_size=args.image_size,
        patch_size=args.patch_size,
        ms_num_channel=args.ms_num_channel,   
        pan_num_channel=args.pan_num_channel,
        schedule_type=args.schedule_type,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        args=args,
    )
    