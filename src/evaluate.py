"""Re-evaluate an exported checkpoint on its original held-out training split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageDataset, stratified_split, transform
from .engine import evaluate
from .model import build_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tta', choices=['none', 'flip'], help='Default: saved training config')
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    data = Path(args.data)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if ckpt['full_data']:
        raise ValueError('Full-data model has seen the validation images: cannot report held-out accuracy')
    manifest = json.loads((data / 'manifest.json').read_text())
    if ckpt['archive_sha256'] != manifest['archive_sha256']:
        raise ValueError('Dataset mismatch')
    labels = np.load(data / 'labels.npy')
    images = np.load(data / 'train_images.npy', mmap_mode='r')
    _, indices = stratified_split(labels, ckpt['config']['val_fraction'], ckpt['config']['split_seed'])
    dataset = ImageDataset(images, indices, labels, transform(ckpt['mean'], ckpt['std']))
    loader = DataLoader(dataset, batch_size=256, num_workers=args.workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = build_model(ckpt['config']).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt['state_dict'])
    result = evaluate(model, loader, device, args.tta or ckpt['config']['tta'])
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
