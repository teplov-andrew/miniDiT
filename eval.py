import hydra
import torch
import os
import wandb
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import make_grid, save_image
from diffusers import AutoencoderKL

from model.dit import DiT
from model.diffusion import DiffusionModel
from utils import show_image_grid


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    if cfg.wandb.logging:
        wandb.init(
            project=cfg.wandb.project_name,
            name=f"{cfg.wandb.run_name}_eval" if cfg.wandb.run_name else "evaluation",
            config=OmegaConf.to_container(cfg, resolve=True),
            job_type="evaluation"
        )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    print("Initializing DiT model...")
    dit = DiT(
        image_size=cfg.model.latent_size, 
        channels_in=cfg.model.channels_in, 
        patch_size=cfg.model.patch_size, 
        hidden_size=cfg.model.hidden_size, 
        num_features=cfg.model.num_features,
        num_layers=cfg.model.num_layers, 
        num_heads=cfg.model.num_heads
    )
    
    print("Initializing Diffusion model...")
    model = DiffusionModel(
        eps_model=dit,
        betas=tuple(cfg.diffusion.betas),
        num_timesteps=cfg.diffusion.num_timesteps,
    ).to(device)
    
    checkpoint_path = cfg.train.checkpoint_path
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully!")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using randomly initialized model.")
    
    model.eval()
    
    num_samples = cfg.eval.num_samples
    latent_hw = (cfg.model.latent_size, cfg.model.latent_size)
    C_lat = cfg.model.channels_in
    
    print(f"Generating {num_samples} samples...")
    
    with torch.no_grad():
        z0 = model.sample(num_samples, (C_lat, *latent_hw), device=device)
        
        z_dec = z0 / 0.18215  # SD scaling factor
        imgs = vae.decode(z_dec).sample
        
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        
        output_path = cfg.eval.sample_images_path
        grid = make_grid(imgs.cpu(), nrow=4, padding=2, normalize=True)
        save_image(grid, output_path)
        print(f"Saved generated samples to {output_path}")
        
        if cfg.wandb.logging:
            wandb.log({
                "samples": wandb.Image(grid, caption=f"Generated Samples")
            })
            
            wandb.finish()
            
        # show_image_grid(imgs, nrow=4, title="Generated Samples")
    
    print("Evaluation completed!")
        


if __name__ == "__main__":
    main()
    