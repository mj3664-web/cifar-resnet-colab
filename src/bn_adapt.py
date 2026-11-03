"""Opt-in unlabeled test BN recalibration; use only with organizer permission."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ImageDataset, digest, transform
from .engine import save_atomic
from .model import build_model


@torch.no_grad()
def recalibrate(model, loader, device, include_flip=True):
    """Reset BN buffers, average batch statistics in one pass, then freeze again.

    Only BN layers enter training mode. Dropout and every other layer stay in eval.
    The returned model uses fixed adapted statistics for subsequent prediction.
    """
    model.eval()
    layers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not layers or any(not m.track_running_stats for m in layers):
        raise ValueError('Expected batch normalization with saved running statistics')
    momenta = [m.momentum for m in layers]
    samples = batches = 0
    try:
        for m in layers:
            m.reset_running_stats()
            m.momentum = None  # Cumulative average of calibration-batch statistics.
            m.train()
        for images, _ in loader:  # Second item is an image index, never a test label.
            samples += len(images)
            if include_flip:
                images = torch.cat((images, images.flip(-1)), dim=0)
            images = images.to(device).contiguous(memory_format=torch.channels_last)
            model(images)
            batches += 1
    finally:
        for m, momentum in zip(layers, momenta):
            m.momentum = momentum
        model.eval()
    if not batches:
        raise ValueError('Empty calibration set')
    return {'images': samples, 'batches': batches, 'bn_layers': len(layers),
            'include_horizontal_flips': include_flip,
            'method': 'Reset all BN running statistics; cumulative average over one shuffled pass'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True, help='Original exported final.pt or best.pt')
    p.add_argument('--out', required=True, help='New checkpoint filename; never overwrite the input')
    p.add_argument('--batch-size', type=int, default=100, help='Source images per calibration batch')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--allow-test-bn-adaptation', action='store_true')
    args = p.parse_args()
    if not args.allow_test_bn_adaptation:
        p.error('Organizer permission is required: pass --allow-test-bn-adaptation only when allowed')
    source, output, data = Path(args.checkpoint), Path(args.out), Path(args.data)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Choose a new output checkpoint path; input and existing outputs are preserved')
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if checkpoint.get('bn_adaptation'):
        raise ValueError('Start with the original unadapted checkpoint')
    manifest = json.loads((data / 'manifest.json').read_text())
    if checkpoint['archive_sha256'] != manifest['archive_sha256']:
        raise ValueError('Archive differs from the checkpoint training archive')
    images = np.load(data / 'test_images.npy', mmap_mode='r')
    # Equal-sized batches keep cumulative averaging from overweighting a short last batch.
    if args.batch_size < 2 or len(images) % args.batch_size:
        raise ValueError('Calibration batch size must be >=2 and divide the test image count exactly')
    dataset = ImageDataset(images, np.arange(len(images)), None,
                           transform(checkpoint['mean'], checkpoint['std']))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=generator, num_workers=args.workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = build_model(checkpoint['config']).to(device).to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    count = sum(v.numel() for v in model.parameters() if v.requires_grad)
    if count != checkpoint['parameter_count']:
        raise ValueError('Checkpoint parameter count mismatch')
    report = recalibrate(model, loader, device, checkpoint['config']['tta'] == 'flip')
    adapted = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    changed = []
    for key, value in adapted.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f'Nonfinite adapted state: {key}')
        if not torch.equal(value, checkpoint['state_dict'][key]):
            if not key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                raise AssertionError(f'A learned parameter unexpectedly changed: {key}')
            changed.append(key)
    report.update(seed=args.seed, calibration_batch_size=args.batch_size,
                  source_checkpoint_sha256=digest(source), trainable_parameters_unchanged=True,
                  parameter_count=count, changed_buffers=changed,
                  scope='Unlabeled competition test images; user confirmed organizer permission',
                  torch_version=str(torch.__version__), device=str(device))
    output.parent.mkdir(parents=True, exist_ok=True)
    save_atomic({**checkpoint, 'state_dict': adapted, 'bn_adaptation': report}, output)
    output.with_suffix('.adaptation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
