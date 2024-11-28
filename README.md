# AVAI_Diffusion
This model aims to replicate the SR3 model for image super-resolution via iterative refinement: https://arxiv.org/abs/2104.07636  
Dataset: DIV2K Train/Valid HR and LR (8x bicubic)
https://data.vision.ee.ethz.ch/cvl/DIV2K/

# Training
Trained on BC4  
```python train1.py --crop-size 128 --epochs 2000 --checkpoint-dir {} --data-dir {}``` 

# Inference
```python inference.py --checkpoint-file {} --crop-size 128 --option "single" --noise-level 0 --down-factor 8 --data-dir {}```  
option: "single" = one batch / "multiple" = entire test set  
noise-level: 0 (no noise), 1 (little), 2 (moderate), 3 (strong)  
down-factor: 8x / 16x  

