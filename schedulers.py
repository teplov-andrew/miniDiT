import math
from torch.optim.lr_scheduler import LambdaLR

def build_warmup_cosine(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=lr_lambda)