import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.utils as vutils

def check_existing_latents(latent_path: str, min_files: int = 1) -> bool:
    if not os.path.exists(latent_path):
        print(f"Latent directory {latent_path} does not exist.")
        return False
    
    pickle_files = glob.glob(os.path.join(latent_path, "*.pkl"))
    num_files = len(pickle_files)
    
    if num_files >= min_files:
        print(f"Found {num_files} existing latent files in {latent_path}. Skipping encoding.")
        return True
    else:
        print(f"Found only {num_files} latent files in {latent_path}. Need at least {min_files}. Will encode images.")
        return False

def load_latents(latent_path):
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, '*.pkl')):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps


def show_image_grid(batch, nrow=4, title=None):
    if batch.min() < 0:
        batch = (batch + 1) / 2.0
    grid = vutils.make_grid(batch, nrow=nrow, padding=2)
    npimg = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(nrow*2, (len(batch)//nrow+1)*2))
    if title:
        plt.title(title)
    plt.imshow(npimg)
    plt.axis("off")
    plt.show()
    
def extract_patches(image_tensor, patch_size=8):
    b, c, h, w = image_tensor.size()

    unfold = nn.Unfold(kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    patches = unfold(image_tensor)
    patches = patches.transpose(1, 2)
    return patches

def reconstruct_image(patch_seq, image_shape, patch_size: int):
    B, C, H, W = image_shape
    L_expected = (H // patch_size) * (W // patch_size)
    fold = nn.Fold(output_size=(H, W), kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))
    patches = patch_seq.transpose(1, 2)
    img = fold(patches)
    return img