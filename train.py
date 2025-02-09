import os
from copy import deepcopy
import time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision as tv
import einops
import matplotlib.pyplot as plt

from diffusion.diffusion_ddpm_pan import GaussianDiffusion
from diffusion.diffusion_ddpm_pan import make_beta_schedule
from models.sr3_dwt import UNetSR3 as Unet
from data.PanDataset import PanDataset
from utils.optim_utils import EmaUpdater
from utils.lr_scheduler import get_lr_from_optimizer, StepsAll
from utils.logger import TensorboardLogger
from utils.misc import grad_clip, model_load
from utils.metric import AnalysisPanAcc
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
def train(
    train_dataset_folder,
    valid_dataset_folder,
    # image settings
    ms_num_channel = 3,
    pan_num_channel = 1,
    image_size=128,
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
    denoise_fn = Unet(
        in_channel=ms_num_channel,
        out_channel=ms_num_channel,
        lms_channel=ms_num_channel,
        pan_channel=pan_num_channel,
        inner_channel=32,
        norm_groups=1,
        channel_mults=(1,2,2,4),#(1, 2, 2, 4),  # (64, 32, 16, 8)
        attn_res=(8,),
        dropout=0.2,
        image_size=image_size,
        self_condition=True,
    ).to(device)
    
    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=image_size,
        channels=ms_num_channel,
        pred_mode="x_start",
        loss_type="l1",
        device=device,
        clamp_range=(0, 1),
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
    opt_d = torch.optim.AdamW(denoise_fn.parameters(), lr=lr_d, weight_decay=1e-4)

    scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
        opt_d, milestones=[100_000, 200_000, 350_000], gamma=0.2
    )
    schedulers = StepsAll(scheduler_d)
    
    train_dataset = PanDataset(
        ms_folder=os.path.join(train_dataset_folder, 'ms_LR'),
        pan_folder=os.path.join(train_dataset_folder, 'pan'),
        GT_folder=os.path.join(train_dataset_folder, 'ms_HR'),
    )
    valid_dataset = PanDataset(
        ms_folder=os.path.join(valid_dataset_folder, 'ms_LR'),
        pan_folder=os.path.join(valid_dataset_folder, 'pan'),
        GT_folder=os.path.join(valid_dataset_folder, 'ms_HR'),
        mode='test',
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    stf_time = time.strftime("%m-%d_%H-%M", time.localtime())
    comment = "PanDIT"
    logger = TensorboardLogger(file_logger_name="{}-{}".format(stf_time, comment))
    
    if pretrain_iterations is not None:
        iterations = pretrain_iterations
        logger.print("load previous training with {} iterations".format(iterations))
    else:
        iterations = 0

    while iterations <= max_iterations:
        for i, batch in enumerate(train_loader):
            pan, lms, hr, wavelets = map(lambda x: x.cuda(), (batch['img_pan'], batch['img_ms_up'], batch['GT'], batch['wavelets']))
            cond, _ = einops.pack(
                [
                    lms,
                    pan,
                    F.interpolate(wavelets, size=lms.shape[-1], mode="bilinear"),
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
            grad_clip(diffusion_dp.model.parameters(), mode="norm", value=0.003)

            opt_d.step()
            ema_updater.update(iterations)
            schedulers.step()

            iterations += 1
            logger.print(
                f"[iter {iterations}/{max_iterations}: "
                + f"d_lr {get_lr_from_optimizer(opt_d): .6f}] - "
                + f"denoise loss {diff_loss:.6f} "
            )

            # test predicted sr
            if show_recon and iterations % 1_000 == 0:
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
            if iterations % 20_000 == 0:
                diffusion_dp.model.eval()
                ema_updater.ema_model.model.eval()

                analysis_d = AnalysisPanAcc()
                with torch.no_grad():
                    for i, batch in enumerate(valid_loader):
                        torch.cuda.empty_cache()
                        pan, lms, hr, wavelets = map(lambda x: x.cuda(), (batch['img_pan'], batch['img_ms_up'], batch['GT'], batch['wavelets']))            
                        cond, _ = einops.pack(
                            [
                                lms,
                                pan,
                                F.interpolate(
                                    wavelets, size=lms.shape[-1], mode="bilinear"
                                ),
                            ],
                            "b * h w",
                        )
                        cond = cond.to(torch.float32)
                        sr = ema_updater.ema_model(cond, mode="ddim_sample", section_counts="ddim25")
                        sr = sr + lms
                        sr = sr.clip(0, 1)

                        hr = hr.to(sr.device)
                        analysis_d(hr, sr)
                        
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
                        logger.print(analysis_d.print_str())

                diffusion_dp.model.train()
                setattr(ema_updater.model, "image_size", 64)

                torch.save(
                    ema_updater.on_fly_model_state_dict,
                    f"./checkpoints/diffusion_iter_{iterations}.pth",
                )
                torch.save(
                    ema_updater.ema_model_state_dict,
                    f"./checkpoints/ema_diffusion_iter_{iterations}.pth",
                )
                logger.print("save model")

                logger.log_scalars("diffusion_perf", analysis_d.acc_ave, iterations)
                logger.print("saved performances")

            # log loss
            if iterations % 50 == 0:
                logger.log_scalar("denoised_loss", diff_loss.item(), iterations)

if __name__ == "__main__":
    py_path = os.path.abspath(__file__) 
    file_dir = os.path.dirname(py_path)
    dataset_folder = os.path.join(os.path.dirname(file_dir), 'PanDataset')
    train(
        train_dataset_folder=os.path.join(dataset_folder, 'WV2_data', 'train128'),
        valid_dataset_folder=os.path.join(dataset_folder, 'WV2_data', 'test128'),
        ms_num_channel=3,
        pan_num_channel=1,
        image_size=128,
        schedule_type="cosine",
        n_steps=500,
        max_iterations=400_000,
        batch_size=1,
        lr_d=1e-5,
        show_recon=True,
        pretrain_weight=None,
        pretrain_iterations=None,
    )