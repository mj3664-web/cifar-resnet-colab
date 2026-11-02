"""Only the user-supplied competition archive is accepted as a data source."""
import csv
import hashlib
import io
import json
import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def read_csv_bytes(raw):
    return list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline='')))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare(archive, out):
    """Read named entries directly: never extract arbitrary archive paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as z:
        candidates = [n for n in z.namelist() if n.endswith('train_labels.csv')]
        if len(candidates) != 1:
            raise ValueError('Expected exactly one train_labels.csv')
        prefix = candidates[0][:-len('train_labels.csv')]
        train_raw = z.read(prefix + 'train_labels.csv')
        train = read_csv_bytes(train_raw)
        test = read_csv_bytes(z.read(prefix + 'test.csv'))
        sample_raw = z.read(prefix + 'sample_submission.csv')
        sample = read_csv_bytes(sample_raw)
        for rows, required in [(train, {'id', 'filename', 'label'}),
                               (test, {'id', 'filename'}), (sample, {'id', 'label'})]:
            if not rows or set(rows[0]) != required:
                raise ValueError(f'Invalid CSV schema, expected {required}')
            if len({r['id'] for r in rows}) != len(rows):
                raise ValueError('Duplicate IDs')
        if set(r['id'] for r in test) != set(r['id'] for r in sample):
            raise ValueError('Test and sample ID sets differ')
        labels = np.array([int(r['label']) for r in train], dtype=np.int64)
        if set(labels.tolist()) != set(range(10)):
            raise ValueError('Expected all ten classes, labeled 0 through 9')
        for name, rows in [('train', train), ('test', test)]:
            if len({r['filename'] for r in rows}) != len(rows):
                raise ValueError('Duplicate filenames')
            images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
            for i, row in enumerate(rows):
                filename = row['filename']
                if '/' in filename or '\\' in filename or filename in {'.', '..'}:
                    raise ValueError('Expected a plain image filename')
                with Image.open(io.BytesIO(z.read(prefix + name + '_images/' + filename))) as img:
                    if img.size != (32, 32) or img.mode != 'RGB':
                        raise ValueError(f'Expected RGB 32x32 image: {filename}')
                    images[i] = np.asarray(img)
            np.save(out / f'{name}_images.npy', images)
        np.save(out / 'labels.npy', labels)
        (out / 'sample_submission.csv').write_bytes(sample_raw)
        (out / 'train_labels.csv').write_bytes(train_raw)
        (out / 'test.csv').write_bytes(z.read(prefix + 'test.csv'))
    manifest = {'archive_sha256': digest(archive), 'train_count': len(train),
                'test_count': len(test), 'class_counts': np.bincount(labels, minlength=10).tolist()}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def stratified_split(labels, fraction=0.1, seed=2026):
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        if count >= len(indices):
            raise ValueError('Each class needs at least two examples')
        val.extend(indices[:count])
        train.extend(indices[count:])
    return np.sort(train), np.sort(val)


def normalization(images, indices):
    """Fit channel statistics exclusively on the selected training partition."""
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    pixels = 0
    for start in range(0, len(indices), 512):
        x = images[indices[start:start + 512]].astype(np.float64) / 255
        total += x.sum(axis=(0, 1, 2))
        squares += (x * x).sum(axis=(0, 1, 2))
        pixels += x.shape[0] * x.shape[1] * x.shape[2]
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean ** 2, 1e-12))
    return mean.tolist(), std.tolist()


def transform(mean, std, training=False, magnitude=7):
    ops = []
    if training:
        ops += [T.RandomCrop(32, padding=4, padding_mode='reflect'), T.RandomHorizontalFlip()]
        if magnitude > 0:
            ops += [T.RandAugment(num_ops=2, magnitude=magnitude,
                                 interpolation=InterpolationMode.BILINEAR,
                                 fill=tuple(round(v * 255) for v in mean))]
    return T.Compose(ops + [T.ToTensor(), T.Normalize(mean, std)])


class ImageDataset(Dataset):
    def __init__(self, images, indices, labels, transform):
        self.images, self.indices, self.labels, self.transform = images, indices, labels, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        x = self.transform(Image.fromarray(self.images[i]))
        return x, (int(self.labels[i]) if self.labels is not None else int(i))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    prepare(args.zip, args.out)
