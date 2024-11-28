import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import io
import os

# Noise level: 0-no noise, 1-little, 2-moderate, 3-strong
# Down factor: 8/16x
class DIV2K(Dataset):
    def __init__(self, hr_dir: str, lr_dir: str, hr_crop_size: int, noise_level: int = 0, down_factor: int = 8):
        assert(hr_crop_size % 8 == 0)
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.noise_level = noise_level
        self.down_factor = down_factor
        assert(self.down_factor % 8 == 0)
        self.hr_crop_size = hr_crop_size
        self.lr_crop_size = hr_crop_size // self.down_factor 
        
        self.hr_image_files = sorted(os.listdir(hr_dir))
        self.lr_image_files = sorted(os.listdir(lr_dir))
        assert(len(self.hr_image_files) == len(self.lr_image_files))

    def __len__(self) -> int:
        return len(self.hr_image_files)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        hr_image_path = os.path.join(self.hr_dir, self.hr_image_files[idx])
        lr_image_path = os.path.join(self.lr_dir, self.lr_image_files[idx])

        hr_image = Image.open(hr_image_path)
        lr_image = Image.open(lr_image_path)
        
        if self.down_factor == 16:
            lr_image = lr_image.resize((lr_image.size[0]//2,lr_image.size[1]//2))
        
        hr_image, lr_image = self.random_crop(hr_image, lr_image, self.hr_crop_size//2, self.hr_crop_size//2)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5,)),
        ])
        hr_tensor = transform(hr_image)
        lr_tensor = transform(lr_image)
        
        lr_tensor = self.apply_noise(lr_tensor)
	
        rescaler = transforms.Resize(self.hr_crop_size, interpolation=transforms.InterpolationMode.BICUBIC)
        lr_tensor_upscaled = rescaler(lr_tensor)
        
        assert(hr_tensor.shape==lr_tensor_upscaled.shape)

        return lr_tensor_upscaled, hr_tensor

    def random_crop(self, hr_image: Image, lr_image: Image, hr_y: int, hr_x: int) -> (Image, Image):
        hr_width, hr_height = hr_image.size
        lr_width, lr_height = lr_image.size
        
        lr_x = hr_x // self.down_factor
        lr_y = hr_y // self.down_factor

        hr_image_cropped = hr_image.crop((hr_x, hr_y, hr_x + self.hr_crop_size, hr_y + self.hr_crop_size))
        lr_image_cropped = lr_image.crop((lr_x, lr_y, lr_x + self.lr_crop_size, lr_y + self.lr_crop_size))
        
        return hr_image_cropped, lr_image_cropped
    
    def apply_noise(self, lr_tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(lr_tensor)
        scale = self.noise_level * 0.02
        noisy_lr_tensor = lr_tensor + noise * scale
        return noisy_lr_tensor
