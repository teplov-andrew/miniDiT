import pickle
import torch
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoencoderKL
from torch.cuda.amp import autocast


def make_latents(train_loader, device, save_dir="latents_pkl", shard_size=2000):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    counter = 0
    shard_idx = 0
    current_shard = {}

    scaling = 0.18215  # SD scale

    @torch.no_grad()
    def encode_batch(x):
        posterior = vae.encode(x).latent_dist
        z = posterior.sample() 
        z = z * scaling
        return z

    for imgs, names in tqdm(train_loader, desc="Encoding"):
        imgs = imgs.to(device, non_blocking=True)
        with autocast(dtype=torch.float16, device_type="cuda"):
            z = encode_batch(imgs)
        z = z.detach().cpu()

        B = z.size(0)
        for i in range(B):
            name = names[i]
            current_shard[name] = [z[i]]
            counter += 1

            if counter % shard_size == 0:
                shard_path = save_dir / f"latents_{shard_idx:05d}.pkl"
                with open(shard_path, "wb") as f:
                    pickle.dump(current_shard, f)
                print(f"Saved {shard_path} with {len(current_shard)} entries")
                shard_idx += 1
                current_shard = {}

    if len(current_shard) > 0:
        shard_path = save_dir / f"latents_{shard_idx:05d}.pkl"
        with open(shard_path, "wb") as f:
            pickle.dump(current_shard, f)
        print(f"Saved {shard_path} with {len(current_shard)} entries")
    
    print(f"Finished creating latents. Total processed: {counter}")