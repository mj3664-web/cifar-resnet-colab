import copy
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def seed_all(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def save_atomic(obj, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        source = model.state_dict()
        for key, value in self.model.state_dict().items():
            if value.is_floating_point():
                value.lerp_(source[key].detach(), 1 - decay)
            else:
                value.copy_(source[key])


def learning_rate(config, progress):
    warmup = config['warmup_epochs']
    if progress < warmup:
        return config['lr'] * max(progress / max(warmup, 1), 0.01)
    fraction = (progress - warmup) / max(config['epochs'] - warmup, 1)
    fraction = min(max(fraction, 0), 1)
    return config['min_lr'] + 0.5 * (config['lr'] - config['min_lr']) * (1 + math.cos(math.pi * fraction))


def mix_batch(images, targets, config, active=True):
    if not active or np.random.random() >= config['mix_probability']:
        return images, targets, targets, 1.0
    choices = [name for name in ('mixup', 'cutmix') if config[name + '_alpha'] > 0]
    if not choices:
        return images, targets, targets, 1.0
    choice = choices[np.random.randint(len(choices))]
    alpha = config[choice + '_alpha']
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(images), device=images.device)
    if choice == 'mixup':
        images = images * lam + images[perm] * (1 - lam)
    else:
        height, width = images.shape[-2:]
        cut_h, cut_w = int(height * math.sqrt(1 - lam)), int(width * math.sqrt(1 - lam))
        cy, cx = np.random.randint(height), np.random.randint(width)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + (cut_h + 1) // 2, height)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + (cut_w + 1) // 2, width)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (height * width))
    return images, targets, targets[perm], lam


def probabilities(model, images, tta='none'):
    probs = model(images).float().softmax(1)
    if tta == 'flip':
        probs = (probs + model(images.flip(-1)).float().softmax(1)) / 2
    elif tta != 'none':
        raise ValueError(f'Unknown TTA: {tta}')
    return probs


@torch.inference_mode()
def evaluate(model, loader, device, tta='none'):
    model.eval()
    correct, total, loss = 0, 0, 0.0
    confusion = torch.zeros(10, 10, dtype=torch.int64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        # Float32 evaluation is also used by submission generation.
        probs = probabilities(model, images, tta)
        pred = probs.argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.numel()
        loss += F.nll_loss(probs.clamp_min(1e-12).log(), targets, reduction='sum').item()
        confusion += torch.bincount((targets * 10 + pred).cpu(), minlength=100).reshape(10, 10)
    return {'accuracy': correct / total, 'loss': loss / total,
            'correct': correct, 'total': total, 'confusion': confusion.tolist()}
