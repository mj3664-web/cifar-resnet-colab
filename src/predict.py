"""Use one exported model; preserve sample_submission.csv ID strings and order."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ImageDataset, digest, transform
from .engine import probabilities
from .model import build_model


def write_submission(sample_path, test_rows, predictions, output):
    with open(sample_path, newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['id', 'label']:
            raise ValueError('Sample must have exactly id,label columns')
        sample = list(reader)
    if len(test_rows) != len(predictions):
        raise ValueError('Prediction count mismatch')
    mapping = {}
    for row, label in zip(test_rows, predictions):
        if row['id'] in mapping:
            raise ValueError('Duplicate test IDs')
        if not isinstance(label, (int, np.integer)) or not 0 <= int(label) <= 9:
            raise ValueError('Labels must be integers from 0 to 9')
        mapping[row['id']] = int(label)
    ids = [row['id'] for row in sample]
    if len(set(ids)) != len(ids) or set(ids) != set(mapping):
        raise ValueError('Sample and test ID sets differ or contain duplicates')
    output = Path(output)
    if output.name != 'submission.csv':
        raise ValueError('Competition requires filename submission.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'label'])
        writer.writerows((i, mapping[i]) for i in ids)
    with output.open(newline='') as stream:
        written = list(csv.DictReader(stream))
    assert [r['id'] for r in written] == ids
    assert len(written) == len(test_rows)
    return len(written)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', default='submission.csv')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    data = Path(args.data)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest = json.loads((data / 'manifest.json').read_text())
    if checkpoint['archive_sha256'] != manifest['archive_sha256']:
        raise ValueError('Dataset differs from training archive')
    config = checkpoint['config']
    model = build_model(config)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != checkpoint['parameter_count']:
        raise ValueError('Parameter count differs from saved metadata')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    images = np.load(data / 'test_images.npy', mmap_mode='r')
    dataset = ImageDataset(images, np.arange(len(images)), None,
                           transform(checkpoint['mean'], checkpoint['std']))
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    labels = []
    for x, _ in loader:
        x = x.to(device).contiguous(memory_format=torch.channels_last)
        labels.extend(probabilities(model, x, config['tta']).argmax(1).cpu().tolist())
    with (data / 'test.csv').open(newline='') as stream:
        test_rows = list(csv.DictReader(stream))
    rows = write_submission(data / 'sample_submission.csv', test_rows, labels, args.out)
    report = {'checkpoint_sha256': digest(args.checkpoint), 'submission_sha256': digest(args.out),
              'archive_sha256': manifest['archive_sha256'], 'parameter_count': count,
              'rows': rows, 'tta': config['tta'], 'weights': checkpoint['weights'],
              'full_data': checkpoint['full_data'], 'epoch': checkpoint['epoch']}
    if checkpoint.get('bn_adaptation'):
        report['bn_adaptation'] = checkpoint['bn_adaptation']
    Path(args.out).with_suffix('.metadata.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
