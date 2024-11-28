import torch
import torch.nn.functional as F
import lpips
from torch import nn

class Metrics:
    def __init__(self):
        self.loss_fn_alex = lpips.LPIPS(net='alex')

    def calculate_mse(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        criterion = nn.MSELoss()
        return criterion(tensor1, tensor2).item()
    
    def calculate_psnr(self, tensor1: torch.Tensor, tensor2: torch.Tensor, max_pixel_value: int = 1.0):
        assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
        mse = F.mse_loss(tensor1, tensor2, reduction='mean')
        if mse == 0:
            return float('inf')
        psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
        return psnr.item()
    
    
    def calculate_ssim(self, tensor1: torch.Tensor, tensor2: torch.Tensor, window_size: int = 11, sigma: float = 1.5, max_pixel_value: float = 1.0) -> float:
        assert tensor1.shape == tensor2.shape, "Input tensors must have the same shape"
    
    
        def create_gaussian_window(window_size, sigma, channels):
            coords = torch.arange(window_size).float() - window_size // 2
            gaussian = torch.exp(-(coords**2) / (2 * sigma**2))
            gaussian = gaussian / gaussian.sum()
            window = gaussian[:, None] @ gaussian[None, :]
            window = window.unsqueeze(0).unsqueeze(0)
            return window.expand(channels, 1, window_size, window_size)
    
        channels = tensor1.size(1)  # Assumes (N, C, H, W)
        window = create_gaussian_window(window_size, sigma, channels).to(tensor1.device)
    
        mu1 = F.conv2d(tensor1, window, groups=channels, padding=window_size // 2)
        mu2 = F.conv2d(tensor2, window, groups=channels, padding=window_size // 2)
    
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
    
        sigma1_sq = F.conv2d(tensor1 * tensor1, window, groups=channels, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(tensor2 * tensor2, window, groups=channels, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(tensor1 * tensor2, window, groups=channels, padding=window_size // 2) - mu1_mu2
    
        C1 = (0.01 * max_pixel_value) ** 2
        C2 = (0.03 * max_pixel_value) ** 2
    
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
        return ssim_map.mean().item()
    
    def calculate_lpips(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        lpips = self.loss_fn_alex(tensor1.cpu(), tensor2.cpu())
        return torch.mean(lpips, axis=0).item()
        
