# mini Diffusion image Transformer
This is a educational repository where I tried to replicate [Diffusion image Transformer](https://arxiv.org/abs/2212.09748) from scratch. It's a small version of DiT and doesn't look like the original version, but the basic principles are preserved. I use a pre-trained VAE and focus only on the implementation of DiT.

## Structure
```bash
miniDiT/
├── README.md
├── config
│   └── config.yaml   # Main config file
├── dataset
│   └── calebahq.py   # CelebA-HQ dataset loader: images + latents
├── model
│   ├── diffusion.py  # Diffusion loop
│   ├── dit.py        # DiT implementation
│   └── vae.py        # VAE wrapper for encoding/decoding between RGB and latent space
├── requirements.txt
├── schedulers.py     # LR scheduler for training
├── training.py       # Main training script
├── eval.py           # Sampling/evaluation script
└── utils.py          
```

## Quick start
- Download the code:
```
git clone https://github.com/teplov-andrew/miniDiT.git
cd miniDiT
```
- Install dependencies:
```
pip install -r requirements.txt
```
- For setting up on CelebHQ, simply download the images from the official repo of CelebMASK HQ [here](https://github.com/switchablenorms/CelebAMask-HQ?tab=readme-ov-file).
- Train
```
python train.py model.hidden_size=512 model.num_layers=20 optimizer.lr=1e-4 train.epochs=50
```
- Evaluation
```
python eval.py eval.num_samples=8
```
## Conclution
The main goal was to try to implement DiT from scratch myself, to better understand how it works. I think that the goal has been achieved, and with a long training period you get a good result, but this does not exclude that there may be bugs in the code, so I will be glad if you find them and write to me