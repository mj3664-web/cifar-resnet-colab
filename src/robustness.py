"""Fixed diagnostic shifts on held-out TRAINING images; never reads test images.

This is a small custom stress test, not CIFAR-10-C or an estimate of Kaggle accuracy.
Recipe selection remains based on clean validation accuracy in src.select.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader

from .data import stratified_split, transform
from .engine import evaluate
from .model import build_model


SHIFTS = ('clean', 'brightness_0.7', 'contrast_0.6', 'blur_1.0', 'noise_0.05')


class ShiftDataset(Dataset):
    def __init__(self, images, labels, indices, normalize, shift):
        if shift not in SHIFTS:
            raise ValueError(f'Unknown shift: {shift}')
        self.images, self.labels, self.indices = images, labels, indices
        self.normalize, self.shift = normalize, shift

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = int(self.indices[index])
        image = Image.fromarray(self.images[i])
        if self.shift == 'brightness_0.7':
            image = ImageEnhance.Brightness(image).enhance(0.7)
        elif self.shift == 'contrast_0.6':
            image = ImageEnhance.Contrast(image).enhance(0.6)
        elif self.shift == 'blur_1.0':
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif self.shift == 'noise_0.05':
            rng = np.random.default_rng(2026 + i)
            x = self.images[i].astype(np.float32) / 255
            x = np.clip(x + rng.normal(0, 0.05, x.shape), 0, 1)
            image = Image.fromarray(np.round(x * 255).astype(np.uint8))
        return self.normalize(image), int(self.labels[i])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--checkpoints', required=True, nargs='+')
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    data = Path(args.data)
    manifest = json.loads((data / 'manifest.json').read_text())
    images = np.load(data / 'train_images.npy', mmap_mode='r')
    labels = np.load(data / 'labels.npy')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    results, signature = [], None
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        if checkpoint['full_data']:
            raise ValueError('Full-data checkpoints have seen the held-out images and cannot be compared here')
        if checkpoint['archive_sha256'] != manifest['archive_sha256']:
            raise ValueError('Archive mismatch')
        current = (config['split_seed'], config['val_fraction'], config['tta'])
        if signature is not None and current != signature:
            raise ValueError('All checkpoints must share split and inference settings')
        signature = current
        _, indices = stratified_split(labels, config['val_fraction'], config['split_seed'])
        model = build_model(config).to(device).to(memory_format=torch.channels_last)
        model.load_state_dict(checkpoint['state_dict'])
        record = {'checkpoint': str(path), 'weights': checkpoint['weights'],
                  'epoch': checkpoint['epoch'], 'parameter_count': checkpoint['parameter_count'],
                  'tta': config['tta'], 'validation_count': len(indices), 'accuracy': {}}
        for shift in SHIFTS:
            dataset = ShiftDataset(images, labels, indices, transform(checkpoint['mean'], checkpoint['std']), shift)
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
            metrics = evaluate(model, loader, device, config['tta'])
            record['accuracy'][shift] = metrics['accuracy']
            print(json.dumps({'checkpoint': str(path), 'shift': shift, 'accuracy': metrics['accuracy']}), flush=True)
        record['shifted_mean_accuracy'] = sum(record['accuracy'][s] for s in SHIFTS[1:]) / (len(SHIFTS) - 1)
        results.append(record)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'description': __doc__, 'results': results}, indent=2))


if __name__ == '__main__':
    main()
