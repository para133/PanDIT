import os
from copy import deepcopy
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision as tv
import einops
import matplotlib.pyplot as plt
import argparse
from itertools import chain

from diffusion.diffusion_ddpm_pan import GaussianDiffusion
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from models.PanVIT import PanDiT
from data.PanDataset import PanDataset
from utils.optim_utils import EmaUpdater
from utils.lr_scheduler import get_lr_from_optimizer, StepsAll
from utils.logger import TensorboardLogger
from utils.misc import grad_clip, model_load
from utils.metrics import AnalysisMetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
def train(
    train_dataset_folder,
    valid_dataset_folder,
    args,
    dataset_name="",
    # image settings
    ms_num_channel = 3,
    pan_num_channel = 1,
    image_size=128,
    patch_size=8,
    # diffusion settings
    schedule_type="cosine",
    n_steps=3_000,
    max_iterations=400_000,
    # optimizer settings
    batch_size=128,
    lr_d=1e-5,
    show_recon=False,
    # pretrain settings
    pretrain_weight=None,
    pretrain_iterations=None,
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
        with_noise_level_emb=args.with_noise_level_emb,
        self_condition=args.self_condition,
    ).to(device)

    
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=ms_num_channel,
        pred_mode="x_start",
        # pred_mode="noise",
        loss_type=args.loss_type,
        device=device,
        clamp_range=(0, 1),
        self_condition=args.self_condition,
    )
    diffusion.set_new_noise_schedule(
        betas=make_beta_schedule(schedule=schedule_type, n_timestep=n_steps, cosine_s=8e-3)
    )
    diffusion = diffusion.to(device)
    
    if pretrain_weight is not None:
        if isinstance(pretrain_weight, (list, tuple)):
            model_load(pretrain_weight[0], denoise_fn, strict=True, device=device)
        else:
            model_load(pretrain_weight, denoise_fn, strict=False, device=device)
        print("load pretrain weight from {}".format(pretrain_weight))
        
    # model, optimizer and lr scheduler
    diffusion_dp = (
        diffusion
    )
    ema_updater = EmaUpdater(
        diffusion_dp, deepcopy(diffusion_dp), decay=0.995, start_iter=20_000
    )
    
    opt_d = torch.optim.AdamW(denoise_fn.parameters(), lr=lr_d, weight_decay=args.weight_decay)

    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[100_000, 200_000, 300_000], gamma=0.2
    )
    schedulers = StepsAll(scheduler_d)
    
    train_dataset = PanDataset(
        img_size=image_size,
        ms_folder=os.path.join(train_dataset_folder, 'ms'),
        pan_folder=os.path.join(train_dataset_folder, 'pan'),
        ms_is_GT=True,
        mode='train',
        start_iter=args.pretrain_iterations,
    )
    valid_dataset = PanDataset(
        img_size=image_size,
        ms_folder=os.path.join(valid_dataset_folder, 'ms'),
        pan_folder=os.path.join(valid_dataset_folder, 'pan'),
        ms_is_GT=True,
        mode='test',
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    stf_time = time.strftime("%m-%d_%H-%M", time.localtime())
    comment = f"{dataset_name}_P{patch_size}C{args.inner_channel}L{args.num_dit_layers}H{args.num_heads}B{batch_size}-{args.model_name}"
    logger = TensorboardLogger(
        place="./runs",
        file_dir="./logs",
        file_logger_name="{}-{}".format(stf_time, comment),
        random_id=False,
        tb_comment="{}-{}".format(stf_time, comment),
    )
    
    if pretrain_iterations != 0:
        iterations = pretrain_iterations
        logger.print("load previous training with {} iterations".format(iterations))
        schedulers.step(iterations)
    else:
        iterations = 0

    while iterations <= max_iterations:
        for i, batch in enumerate(train_loader):
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
            opt_d.zero_grad()
            res = hr - lms
            diff_loss, recon_x = diffusion_dp(res, cond=cond)
            diff_loss.backward()
            recon_x = recon_x + lms

            # do a grad clip on diffusion model
            grad_clip(diffusion_dp.model.parameters(), mode="norm", value=1.0)

            opt_d.step()
            ema_updater.update(iterations)
            schedulers.step()

            iterations += 1
            logger.print(
                f"[iter {iterations}/{max_iterations}: "
                + f"d_lr {get_lr_from_optimizer(opt_d): .8f}] - "
                + f"denoise loss {diff_loss:.6f} "
            )

            # test predicted sr
            if show_recon and iterations % 5_000 == 0:
                # NOTE: only used to validate code
                recon_x = recon_x[:64]

                x = tv.utils.make_grid(recon_x, nrow=8, padding=0).cpu()
                x = x.clip(0, 1)  # for no warning
                fig, ax = plt.subplots(figsize=(x.shape[-1] // 100, x.shape[-2] // 100))
                x_show = (
                    x.permute(1, 2, 0).detach().numpy()
                )
                ax.imshow(x_show)
                ax.set_axis_off()
                plt.tight_layout(pad=0)
                fig.savefig(
                    f"./samples/recon_x/iter_{iterations}.png",
                    dpi=200,
                    bbox_inches="tight",
                    pad_inches=0,
                )

            # do some sampling to check quality
            if iterations % args.save_per_iter == 0:
                diffusion_dp.model.eval()
                ema_updater.ema_model.model.eval()

                analysis_metrics = AnalysisMetrics()
                with torch.no_grad():
                    for i, batch in enumerate(valid_loader):
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
                        sr = ema_updater.ema_model(cond, mode="ddim_sample", section_counts="ddim25")
                        # sr = ema_updater.ema_model(cond, mode="ddpm_sample")
                        sr = sr + lms
                        sr = sr.clip(0, 1)
                        hr = hr.to(sr.device)
                        analysis_metrics.batch_metrics(sr, hr, mode=args.mode)
                        
                        hr = tv.utils.make_grid(hr, nrow=4, padding=0).cpu()
                        x = tv.utils.make_grid(sr, nrow=4, padding=0).detach().cpu()
                        x = x.clip(0, 1)

                        s = torch.cat([hr, x], dim=-1)  # [b, c, h, 2*w]
                        fig, ax = plt.subplots(
                            figsize=(s.shape[-1] // 100, s.shape[-2] // 100)
                        )
                        ax.imshow(
                            s.permute(1, 2, 0).detach().numpy()
                        )
                        ax.set_axis_off()

                        plt.tight_layout(pad=0)
                        fig.savefig(
                            f"./samples/valid_samples/iter_{iterations}.png",
                            dpi=200,
                            bbox_inches="tight",
                            pad_inches=0,
                        )
                    logger.print("---diffusion result---")
                    logger.print(analysis_metrics.get_metrics_str())

                diffusion_dp.model.train()

                torch.save(
                    ema_updater.on_fly_model_state_dict,
                    f"./checkpoints/diffusion_{comment}_iter_{iterations}.pth",
                )
                torch.save(
                    ema_updater.ema_model_state_dict,
                    f"./checkpoints/ema_diffusion_{comment}_iter_{iterations}.pth",
                )
                logger.print("save model")

                logger.log_scalars("diffusion_perf", analysis_metrics.get_metrics(), iterations)
                logger.print("saved performances")

            # log loss
            if iterations % 500 == 0:
                logger.log_scalar("denoised_loss", diff_loss.item(), iterations)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for PanDiT")
    # parser.add_argument('--train_dataset_folder', type=str, required=True, help='Path to the training dataset folder')
    # parser.add_argument('--valid_dataset_folder', type=str, required=True, help='Path to the validation dataset folder')
    parser.add_argument('--model_name', type=str, default='PanDiT', help='Model name')
    parser.add_argument('--dataset_name', type=str, default='WV2', help='Dataset name')
    parser.add_argument('--ms_num_channel', type=int, default=4, help='Number of multispectral channels')
    parser.add_argument('--pan_num_channel', type=int, default=1, help='Number of panchromatic channels')
    parser.add_argument('--image_size', type=int, default=128, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=8, help='Patch size for processing')
    parser.add_argument('--schedule_type', type=str, default='cosine', choices=['cosine', 'linear'], help='Schedule type for beta schedule')
    parser.add_argument('--n_steps', type=int, default=500, help='Number of diffusion steps')
    parser.add_argument('--max_iterations', type=int, default=200_000, help='Maximum number of training iterations')
    parser.add_argument('--save_per_iter', type=int, default=20_000, help='Save model per iteration')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--lr_d', type=float, default=5e-5, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1sam', 'l1', 'l1ssim'], help='Loss type for diffusion')
    parser.add_argument('--show_recon', type=bool, default=True, help='Whether to show reconstructed images during training')
    parser.add_argument('--pretrain_weight', type=str, default=None, help='Path to pretrained weights (if any)')
    parser.add_argument('--pretrain_iterations', type=int, default=0, help='Number of iterations for pretrained model')
    parser.add_argument('--inner_channel', type=int, default=16, help='Number of inner channels')
    parser.add_argument('--noise_level_channel', type=int, default=128, help='Number of noise level channels')
    parser.add_argument('--time_hidden_ratio', type=int, default=4, help='Time hidden ratio')
    parser.add_argument('--num_dit_layers', type=int, default=12, help='Number of DIT layers')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='MLP ratio in DIT Block')
    parser.add_argument('--with_noise_level_emb', type=bool, default=True, help='Whether to include noise level embedding')
    parser.add_argument('--self_condition', type=bool, default=True, help='Whether to self condition')
    parser.add_argument('--mode', type=str, default='CMYK', help='Path to save the output images')
    return parser.parse_args()

if __name__ == "__main__":
    py_path = os.path.abspath(__file__) 
    file_dir = os.path.dirname(py_path)
    dataset_folder = os.path.join(os.path.dirname(file_dir), 'PanDataset')
    args = parse_args()
    # args.pretrain_weight = os.path.join(file_dir, 'checkpoints', 'diffusion_Best_WV3_Pyramid_P8C16L12H16B6-PanDiT_iter_140000.pth')
    # args.pretrain_iterations = 140_000
    # args.save_per_iter = 5000
    train(
        train_dataset_folder=os.path.join(dataset_folder, 'WV2_data', 'train128'),
        valid_dataset_folder=os.path.join(dataset_folder, 'WV2_data', 'test128'),
        dataset_name=args.dataset_name,
        ms_num_channel=args.ms_num_channel,
        pan_num_channel=args.pan_num_channel,
        patch_size=args.patch_size,
        image_size=args.image_size,
        schedule_type=args.schedule_type,
        n_steps=args.n_steps,
        max_iterations=args.max_iterations,
        batch_size=args.batch_size,
        lr_d=args.lr_d,
        show_recon=args.show_recon,
        pretrain_weight=args.pretrain_weight,
        pretrain_iterations=args.pretrain_iterations,
        args=args,
    )