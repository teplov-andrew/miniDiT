import hydra
import torch
import numpy as np
import random
import os
import glob
import wandb
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
import torchvision.transforms as T

from model.vae import make_latents
from model.dit import DiT
from model.diffusion import DiffusionModel
from dataset.calebahq import CelebAHQDataset
from schedulers import build_warmup_cosine
from utils import check_existing_latents

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.train.seed)
    
    if cfg.wandb.logging:
        wandb.init(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = T.Compose([
        T.Resize(128, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])
    
    latent_path = cfg.data.latent_path
    latents_exist = check_existing_latents(latent_path, min_files=1)
    
    if not latents_exist:
        print("Creating latents from images...")
        train_dataset_imgs = CelebAHQDataset(
            cfg.data.img_path, 
            transform=transform, 
            mode="image", 
            return_name=True
            )
        
        train_loader_imgs = DataLoader(
            train_dataset_imgs, 
            batch_size=8, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
            )
        
        make_latents(train_loader_imgs, device, save_dir=latent_path)
    else:
        print("Using existing latents.")
    
    train_dataset_latents = CelebAHQDataset(
        root_dir="data/celeba_hq_128",
        use_latents=True,
        latent_path=cfg.data.latent_path,  
        mode="latent",
        return_name=True     
    )
    
    latents_dataloader = DataLoader(
        train_dataset_latents,
        batch_size=cfg.data.batch_size,
        shuffle=True,                
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    dit = DiT(
        image_size=cfg.model.latent_size, 
        channels_in=cfg.model.channels_in, 
        patch_size=cfg.model.patch_size, 
        hidden_size=cfg.model.hidden_size, 
        num_features=cfg.model.num_features,
        num_layers=cfg.model.num_layers, 
        num_heads=cfg.model.num_heads
    )
    
    model = DiffusionModel(
        eps_model=dit,
        betas=tuple(cfg.diffusion.betas),
        num_timesteps=cfg.diffusion.num_timesteps,
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=cfg.optimizer.lr,
                      betas=tuple(cfg.optimizer.betas),
                      weight_decay=cfg.optimizer.weight_decay)

    total_steps = cfg.train.epochs * max(1, len(latents_dataloader))
    warmup_steps = max(1, int(cfg.train.warmup_ratio * total_steps))
    scheduler = build_warmup_cosine(optimizer, total_steps, warmup_steps)
    
    use_amp = cfg.train.amp and torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    
    loss_lst = []
    global_step = 0
    
    for epoch in tqdm(range(cfg.train.epochs)):
        running = 0.0
        count = 0

        for num_iter, batch in enumerate(latents_dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp, device_type="cuda"):
                latents = batch[0].to(device)
                loss = model(latents)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            bsz = latents.size(0)
            running += loss.item() * bsz
            count   += bsz
            loss_lst.append(loss.item())
            
            
            current_lr = scheduler.get_last_lr()[0]
            global_step += 1

        epoch_loss = running / max(1, count)
        current_lr = scheduler.get_last_lr()[0]
        
        if cfg.wandb.logging:
            wandb.log({
                "epoch/loss": epoch_loss,
                "epoch/loss_mean": np.mean(loss_lst),
                "epoch/lr": current_lr,
                "epoch/epoch": epoch
            }, step=global_step)
        
        print(f"Epoch: {epoch}  Loss (mean): {np.mean(loss_lst)}    Loss (epoch mean): {epoch_loss:.6f}    LR: {current_lr:.2e}")
        
    checkpoint_path = cfg.train.checkpoint_path
    torch.save(model.state_dict(), checkpoint_path)
    
    config_path = "config_used.yaml"
    OmegaConf.save(cfg, config_path)
    
    if cfg.wandb.logging:
        wandb.save(checkpoint_path)
        wandb.save(config_path)
        wandb.finish()
    
if __name__ == "__main__":
    main()