from PIL import Image
import torch
import numpy as np
from dataset import DIV2K
from torch.utils.data import Dataset, DataLoader

def save_image(image_tensor: torch.Tensor, save_path: str):
    image_numpy = image_tensor.permute(1,2,0).cpu().numpy()
    image_numpy = (image_numpy - np.min(image_numpy)) / (np.max(image_numpy) - np.min(image_numpy) + 1e-5)
    image_numpy = image_numpy * 255
    image = Image.fromarray(image_numpy.astype('uint8'))
    image.save(save_path)
    
def save_image_batch(image_tensor_batch: torch.Tensor, save_path:str):
    for batch_index, image_tensor in enumerate(image_tensor_batch):  
        output_path = f"{save_path}-{batch_index}.jpg"
        save_image(image_tensor, output_path)
    
def load_data(data_dir: str, hr_crop_size: int):
    train_dataset = DIV2K(
        hr_dir = f"{data_dir}/DIV2K_train_HR",
        lr_dir = f"{data_dir}/DIV2K_train_LR_x8",
        hr_crop_size = hr_crop_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, num_workers=8)
    
    val_dataset = DIV2K(
        hr_dir = f"{data_dir}/DIV2K_valid_HR",
        lr_dir = f"{data_dir}/DIV2K_valid_LR_x8",
        hr_crop_size = hr_crop_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=8)
    return train_loader, val_loader    