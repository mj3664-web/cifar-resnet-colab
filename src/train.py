"""Train from scratch, validate only on allowed training data, resume by epoch."""
import argparse
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from .data import ImageDataset, normalization, seed_worker, stratified_split, transform
from .engine import EMA, evaluate, learning_rate, mix_batch, save_atomic, seed_all
from .model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true', help='Resume out/last.pt if present')
    parser.add_argument('--full-data', action='store_true', help='Final fit: no validation or checkpoint selection')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Stop at this completed epoch; keep original LR schedule')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if not 0 < config['val_fraction'] < 1 or config['epochs'] < 1:
        raise ValueError('Invalid epoch count or validation fraction')
    if config['batch_size'] < 2 or config['warmup_epochs'] >= config['epochs']:
        raise ValueError('Need batch_size >= 2 and warmup_epochs < epochs')
    if config['inference_weights'] not in ('raw', 'ema') or config['tta'] not in ('none', 'flip'):
        raise ValueError('Invalid inference settings')
    device = torch.device(args.device)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    seed_all(config['seed'], args.deterministic)
    # Keep float32 evaluation comparable between training and standalone inference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out, data = Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / 'last.pt'
    if checkpoint_path.exists() and not args.resume:
        raise ValueError('Run already exists; use --resume or choose a new output directory')
    manifest = json.loads((data / 'manifest.json').read_text())
    images = np.load(data / 'train_images.npy', mmap_mode='r')
    labels = np.load(data / 'labels.npy')
    train_indices, val_indices = stratified_split(labels, config['val_fraction'], config['split_seed'])
    if args.full_data:
        train_indices, val_indices = np.arange(len(labels)), np.array([], dtype=np.int64)
    mean, std = normalization(images, train_indices)
    np.savez(out / 'split.npz', train_indices=train_indices, val_indices=val_indices)
    model = build_model(config).to(device).to(memory_format=torch.channels_last)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Parameters: {parameter_count:,}; train={len(train_indices)}; val={len(val_indices)}', flush=True)
    ema = EMA(model, config['ema_decay'])
    # Exclude BN scales and biases from weight decay.
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    optimizer = torch.optim.SGD([{'params': decay, 'weight_decay': config['weight_decay']},
                                 {'params': no_decay, 'weight_decay': 0.0}],
                                lr=config['lr'], momentum=config['momentum'], nesterov=True)
    use_amp = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and amp_dtype == torch.float16)
    training = ImageDataset(images, train_indices, labels,
                            transform(mean, std, True, config['randaugment_magnitude']))
    validation = ImageDataset(images, val_indices, labels, transform(mean, std))
    generator = torch.Generator()
    loader = DataLoader(training, batch_size=config['batch_size'], shuffle=True,
                        num_workers=args.workers, pin_memory=use_amp, worker_init_fn=seed_worker,
                        generator=generator, persistent_workers=False, drop_last=False)
    val_loader = (DataLoader(validation, batch_size=config['batch_size'] * 2,
                            num_workers=args.workers, pin_memory=use_amp) if len(val_indices) else None)
    start, best, best_info = 0, -1.0, None
    if args.resume and checkpoint_path.exists():
        # Resume files contain RNG objects. Load only checkpoints created by this repository.
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if state['config'] != config or state['full_data'] != args.full_data:
            raise ValueError('Cannot change configuration or data mode when resuming')
        if state['archive_sha256'] != manifest['archive_sha256']:
            raise ValueError('Dataset archive changed')
        if state['deterministic'] != args.deterministic:
            raise ValueError('Deterministic flag changed')
        model.load_state_dict(state['model'])
        ema.model.load_state_dict(state['ema'])
        ema.updates = state['ema_updates']
        optimizer.load_state_dict(state['optimizer'])
        for item in optimizer.state.values():
            for key, value in item.items():
                if torch.is_tensor(value):
                    item[key] = value.to(device)
        scaler.load_state_dict(state['scaler'])
        start, best, best_info = state['epoch'], state['best'], state['best_info']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        if use_amp and state['rng']['cuda']:
            torch.cuda.set_rng_state_all(state['rng']['cuda'])
        print(f'Resuming after epoch {start}', flush=True)
    metadata = {'config': config, 'parameter_count': parameter_count, 'mean': mean, 'std': std,
                'full_data': args.full_data, 'archive_sha256': manifest['archive_sha256'],
                'train_count': len(train_indices), 'val_count': len(val_indices),
                'deterministic': args.deterministic}
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    environment = {'python': platform.python_version(), 'torch': torch.__version__,
                   'torchvision': torchvision.__version__, 'numpy': np.__version__,
                   'cuda': torch.version.cuda, 'device': str(device),
                   'gpu': torch.cuda.get_device_name(device) if use_amp else None,
                   'amp_dtype': str(amp_dtype) if use_amp else None}
    (out / f'environment_epoch_{start}.json').write_text(json.dumps(environment, indent=2))
    try:
        freeze = subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True)
        (out / f'pip_freeze_epoch_{start}.txt').write_text(freeze)
    except subprocess.CalledProcessError:
        pass
    log_path = out / 'history.jsonl'
    if log_path.exists():
        history = [json.loads(line) for line in log_path.read_text().splitlines() if line]
        log_path.write_text(''.join(json.dumps(row) + '\n' for row in history if row['epoch'] <= start))
    end = min(config['epochs'], args.stop_after or config['epochs'])

    def export(name, weights, epoch, selection):
        chosen = ema.model if weights == 'ema' else model
        save_atomic({**metadata, 'state_dict': chosen.state_dict(), 'epoch': epoch,
                     'weights': weights, 'selection': selection}, out / name)

    for epoch in range(start, end):
        begin = time.time()
        generator.manual_seed(config['seed'] + epoch)
        model.train()
        running_loss, seen = 0.0, 0
        for step, (x, y) in enumerate(loader):
            lr = learning_rate(config, epoch + (step + 1) / len(loader))
            for group in optimizer.param_groups:
                group['lr'] = lr
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            x, ya, yb, lam = mix_batch(x, y, config, epoch < config['epochs'] - config['mix_off_last'])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = lam * F.cross_entropy(logits, ya, label_smoothing=config['label_smoothing'])
                loss += (1 - lam) * F.cross_entropy(logits, yb, label_smoothing=config['label_smoothing'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss; last completed epoch is recoverable')
            scaler.scale(loss).backward()
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                ema.update(model)
            running_loss += loss.item() * len(y)
            seen += len(y)
        record = {'epoch': epoch + 1, 'train_loss': running_loss / seen, 'lr': lr}
        if val_loader is not None:
            for name, network in [('raw', model), ('ema', ema.model)]:
                metrics = evaluate(network, val_loader, device, config['tta'])
                record[name + '_accuracy'] = metrics['accuracy']
                record[name + '_loss'] = metrics['loss']
                if metrics['accuracy'] > best:
                    best = metrics['accuracy']
                    best_info = {'epoch': epoch + 1, 'weights': name, 'tta': config['tta'], **metrics}
                    export('best.pt', name, epoch + 1, best_info)
                    (out / 'best_metrics.json').write_text(json.dumps(best_info, indent=2))
        record['seconds'] = time.time() - begin
        record['best_accuracy'] = best if best_info else None
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if use_amp else []}
        save_atomic({**metadata, 'model': model.state_dict(), 'ema': ema.model.state_dict(),
                     'ema_updates': ema.updates, 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'best': best,
                     'best_info': best_info, 'rng': rng}, checkpoint_path)
        with log_path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if epoch + 1 == config['epochs']:
            export('final.pt', config['inference_weights'], epoch + 1,
                   'Fixed final epoch; no test/validation selection' if args.full_data else best_info)
    # Recover an interruption between saving the last epoch and writing the export.
    if start >= config['epochs'] and not (out / 'final.pt').exists():
        export('final.pt', config['inference_weights'], start,
               'Fixed final epoch; no test/validation selection' if args.full_data else best_info)
    print(f'Completed {end}/{config["epochs"]} epochs. Checkpoints: {out}', flush=True)


if __name__ == '__main__':
    main()
