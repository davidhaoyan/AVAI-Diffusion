from dataset import DIV2K
from train1 import Diffusion, evaluate_model
import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torch import nn
import os
from time import time
import argparse
from metrics import Metrics
from helper import save_image_batch
from datetime import datetime

parser = argparse.ArgumentParser(
    description="Inference SR Diffusion",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--checkpoint-file",
    default="/user/work/lj21689/avai/dif/checkpoints4/run_2024-11-24_17-25-38/checkpoint_epoch_19_2024-11-24_17-30-35.pth.tar",
    type=str,
)
parser.add_argument(
    "--crop-size",
    default=128,
    type=int,
)
parser.add_argument(
    "--option",
    default="multiple",
    type=str, 
    help="single stops after one batch, multiple runs through all batches"
)
parser.add_argument(
    "--noise-level",
    default=0,
    type=int,
    help="degrades the test set, {0-no noise, 1-little, 2-moderate, 3-high}"
)
parser.add_argument(
    "--down-factor",
    default=8,
    type=int,
    help="8/16x downsampling"
)
parser.add_argument(
    "--notes",
    default="",
    type=str,
)
parser.add_argument(
    "--data-dir",
    default="/home/lj21689/avai/dip",
    type=str,
)


class Inferencer():
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        checkpoint_file: str,
        metrics: Metrics,
        output_dir: str,
    ):
        self.model=model
        self.test_loader=test_loader
        self.device=device
        if checkpoint_file != "":
            self.load_checkpoint(checkpoint_file)
        self.metrics = metrics
        self.output_dir = output_dir
            
    def load_checkpoint(self, checkpoint_file:str):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_file}")
    
    def infer(self, option="single"):
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        for index, (low_res, high_res) in enumerate(self.test_loader):
            original_images_path = f"{self.output_dir}/original"
            save_image_batch(low_res, f"{original_images_path}_low-{index}")
            save_image_batch(high_res, f"{original_images_path}_high-{index}")

            mse, psnr, ssim, lpips = evaluate_model(
                model=self.model, 
                device=self.device, 
                low_res=low_res, 
                high_res=high_res, 
                metrics=self.metrics, 
                save_path=f"{self.output_dir}/image-{index}"
            )
            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
            if option=="single": 
                print(f"Test MSE: {mse} PSNR: {psnr} SSIM: {ssim} LPIPS: {lpips}")
                break
        avg_mse = total_mse / len(self.test_loader)
        avg_psnr = total_psnr / len(self.test_loader)
        avg_ssim = total_ssim / len(self.test_loader)
        avg_lpips = total_lpips / len(self.test_loader)
        print(f"Test Average MSE: {mse} Average PSNR: {psnr} Average SSIM: {ssim} Average LPIPS: {lpips}")
 
        
def main(args): 
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    data_dir = args.data_dir
    val_dataset = DIV2K(
        hr_dir = f"{data_dir}/DIV2K_valid_HR",
        lr_dir = f"{data_dir}/DIV2K_valid_LR_x8",
        hr_crop_size= args.crop_size,
        noise_level=args.noise_level,
        down_factor=args.down_factor,
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=8)

    model = Diffusion(
        n_times=1000, 
        beta_minmax=[1e-4, 2e-2], 
        image_channels=3,
    )
    model.to(DEVICE)

    metrics = Metrics()
    output_dir = f"./inference/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_noise_{args.noise_level}_down_{args.down_factor}"
    os.makedirs(output_dir, exist_ok=True)
    
    inferencer = Inferencer(
        model=model, 
        test_loader=val_loader, 
        device=DEVICE, 
        checkpoint_file=args.checkpoint_file, 
        metrics=metrics,
        output_dir=output_dir
    )
    inferencer.infer(option=args.option)
        
if __name__ == "__main__":
    main(parser.parse_args())
        
