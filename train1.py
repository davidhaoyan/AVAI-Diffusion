import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import DIV2K
from time import time
import numpy as np
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
import argparse
from multiprocessing import cpu_count
import os
from PIL import Image
from datetime import datetime
from helper import save_image_batch, load_data
import lpips
from torchvision import transforms
from metrics import Metrics
from model import UNet

parser = argparse.ArgumentParser(
    description="Training SR Diffusion",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--checkpoint-dir",
    default="/user/work/lj21689/avai/dif/checkpoints",
    type=str,
)
parser.add_argument(
    "--epochs",
    default=20,
    type=int,
)
parser.add_argument(
    "--crop-size",
    default=256,
    type=int,
)
parser.add_argument(
    "--data-dir",
    default="/user/work/lj21689",
    type=str,
)

class Diffusion(nn.Module):
    def __init__(self, n_times: int, beta_minmax: [float], image_channels: int):
        super().__init__()
        self.n_times = n_times
        self.image_channels = image_channels
        self.betas = torch.linspace(beta_minmax[0], beta_minmax[1], self.n_times)
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim = -1)
        self.model = UNet(input_channels = 2*self.image_channels, output_channels = self.image_channels, n_times = self.n_times)

    def add_noise(self, x, timesteps):
        noise = torch.randn_like(x)
        noised_examples = []
        for index, timestep in enumerate(timesteps):
            alpha_hat_t = self.alpha_hats[timestep]
            noised_examples.append(torch.sqrt(alpha_hat_t)*x[index] + torch.sqrt(1 - alpha_hat_t)*noise[index])
        return torch.stack(noised_examples), noise

    def forward(self, x, timestep):
        return self.model(x, timestep)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        summary_writer: SummaryWriter,
        checkpoint_path: str,
        metrics: Metrics
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.checkpoint_path = checkpoint_path
        self.step = 0
        self.metrics = metrics
        
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(), 
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        checkpoint_file = f"{self.checkpoint_path}/checkpoint_epoch_{epoch}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth.tar"
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")

    def load_checkpoint(self, checkpoint_file: str):
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        print(f"Checkpoint loaded from {checkpoint_file}")
        return checkpoint['epoch']  # Return the starting epoch
    
    def train(self, epochs, val_frequency, print_frequency, log_frequency, checkpoint_frequency):
        self.model.model.to(self.device)
        for epoch in range(epochs):
            self.model.model.train()
            print(f"Epoch {epoch}:")
            losses = []
            stime = time()

            for i, (low_res, high_res) in enumerate(self.train_loader):
                batches = high_res.shape[0]
                low_res, high_res = low_res.to(self.device), high_res.to(self.device)

                timesteps = torch.randint(low = 1, high = self.model.n_times, size = (batches, ))
                gamma = self.model.alpha_hats[timesteps].to(self.device)
                timesteps = timesteps.to(device = self.device)

                high_res, gt_noise = self.model.add_noise(high_res, timesteps)
                high_res = torch.cat([low_res, high_res], dim = 1) #condition on low_res

                predicted_noise = self.model(high_res, gamma)
                loss = self.criterion(gt_noise, predicted_noise)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

                if i % print_frequency == 0:
                    self.print_metrics(epoch, loss)
                self.step += 1

            ftime = time()
            print(f"Epoch trained in {ftime - stime}s; Avg loss => {sum(losses)/len(losses)}")
            
            if epoch % checkpoint_frequency == 0:
                self.save_checkpoint(epoch)
            if epoch % log_frequency == 0:
                self.log_metrics(epoch, loss)
            if epoch % val_frequency == 0:
                self.validate(epoch)
                
            #torch.save(self.model.model.state_dict(), f"{checkpoint_dir}/sr_ep_{ep}.pt")
         
        print("Training finished")
        
    def print_metrics(self, epoch, loss):
        epoch_step = (self.step % len(self.train_loader)) + 1
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
        )
        
    def log_metrics(self, epoch, loss):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
           
    def validate(self, epoch: int):
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        for low_res, high_res in self.val_loader:
            mse, psnr, ssim, lpips = evaluate_model(self.model, self.device, low_res, high_res, self.metrics)
            total_mse += mse
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips
        avg_mse = total_mse / len(self.val_loader)
        avg_psnr = total_psnr / len(self.val_loader)
        avg_ssim = total_ssim / len(self.val_loader)
        avg_lpips = total_lpips / len(self.val_loader)
        self.summary_writer.add_scalars(
            "mse",
            {"train": float(avg_mse)},
            self.step
        )
        self.summary_writer.add_scalars(
            "psnr",
            {"train": float(avg_psnr)},
            self.step
        )        
        self.summary_writer.add_scalars(
            "ssim",
            {"train": float(avg_ssim)},
            self.step
        )
        self.summary_writer.add_scalars(
            "lpips",
            {"train": float(avg_lpips)},
            self.step
        )
        print(f"Epoch {epoch}: Validation MSE: {avg_mse} PSNR: {avg_psnr} SSIM: {avg_ssim} LPIPS: {avg_lpips}")

def sample(model: nn.Module, device: torch.device, low_res: torch.Tensor) -> torch.Tensor:
    model.model.to(device)
    model.model.eval()

    stime = time()
    with torch.no_grad():

        y = torch.randn_like(low_res, device = device)
        low_res = low_res.to(device)
        for i, t in enumerate(range(model.n_times - 1, 0 , -1)):
            alpha_t, alpha_t_hat, beta_t = model.alphas[t], model.alpha_hats[t], model.betas[t]

            t = torch.tensor(t, device = device).long()
            pred_noise = model(torch.cat([low_res, y], dim = 1), alpha_t_hat.view(-1).to(device))
            y = (torch.sqrt(1/alpha_t))*(y - (1-alpha_t)/torch.sqrt(1 - alpha_t_hat) * pred_noise)
            if t > 1:
                noise = torch.randn_like(y)
                y = y + torch.sqrt(beta_t) * noise

    ftime = time()
    print(f"Done denoising in {ftime - stime}s ")
    return y
        
def evaluate_model(
    model: nn.Module, 
    device: torch.device, 
    low_res: torch.Tensor, 
    high_res: torch.Tensor, 
    metrics: Metrics, 
    save_path: str = ""
) -> float:
    # takes in a batch
    prediction = sample(model, device, low_res)
    
    # for Inferencer
    if save_path != "":
        save_image_batch(prediction, save_path)
        
    prediction = prediction.to(device)
    ground_truth = high_res.to(device)

    mse = metrics.calculate_mse(prediction, ground_truth)
    psnr = metrics.calculate_psnr(prediction, ground_truth)
    ssim = metrics.calculate_ssim(prediction, ground_truth)
    lpips = metrics.calculate_lpips(prediction, ground_truth)
    return mse, psnr, ssim, lpips

def main(args):  
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)
    
    train_loader, val_loader = load_data(args.data_dir, args.crop_size)

    model = Diffusion(n_times=1000, beta_minmax=[1e-4, 2e-2], image_channels=3)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.model.parameters(), lr = 1e-3)

    log_dir = "/user/work/lj21689/avai/dif/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_epochs_{args.epochs}" 
    print(f"Writing logs to {log_path}")
    summary_writer = SummaryWriter(
        str(log_path),
        flush_secs=5,
    )
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{args.checkpoint_dir}/run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_epochs_{args.epochs}"
    os.makedirs(checkpoint_path, exist_ok=True)

    metrics = Metrics()
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        summary_writer=summary_writer,
        checkpoint_path=checkpoint_path,
        metrics=metrics
    )

    trainer.train(
        epochs=args.epochs,
        val_frequency=50,
        print_frequency=10,
        log_frequency=2,
        checkpoint_frequency=100,
        #checkpoint_file="/user/work/lj21689/avai/dif/checkpoints/run_2024-11-24_02-22-10/checkpoint_epoch_279_2024-11-24_03-18-27.pth.tars",
    )
        
if __name__ == "__main__":
    main(parser.parse_args())