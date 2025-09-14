from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from utils import load_latents


class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, transform=None, use_latents=False, latent_path=None, mode="image", return_name=False):
        self.root_dir = Path(root_dir)
        self.image_paths = sorted(list(self.root_dir.glob("*.jpg")) + list(self.root_dir.glob("*.png")))
        self.transform = transform
        self.mode = mode
        self.return_name = return_name
        self.use_latents = False

        if use_latents and latent_path is not None:
            lm = load_latents(latent_path)
            self.latent_maps = {Path(k).stem: v for k, v in lm.items()}
            self.latent_keys = sorted(self.latent_maps.keys())
            keys = set(self.latent_maps.keys())
            img_keys = {p.stem for p in self.image_paths}
            if not img_keys.issubset(keys) and self.mode in ("latent", "both"):
                self.image_paths = sorted([p for p in self.image_paths if p.stem in keys])
            print(f"Found {len(self.latent_maps)} latents")
            self.use_latents = True
        else:
            self.latent_maps, self.latent_keys = {}, []

    def __len__(self):
        if self.mode == "latent" and self.use_latents:
            return len(self.latent_keys)
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.mode == "latent":
            key = self.latent_keys[idx]
            latent = self.latent_maps[key]
            name = f"{key}.jpg"
            return (latent, name) if self.return_name else latent

        img_path = self.image_paths[idx]
        img_name = img_path.name
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.mode == "image":
            return (img, img_name) if self.return_name else img
        else:
            key = img_path.stem
            latent = self.latent_maps[key]
            out = {"image": img, "latent": latent, "path": str(img_path)}
            if self.return_name:
                out["name"] = img_name
            return out