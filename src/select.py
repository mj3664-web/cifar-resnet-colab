"""Select a completed experiment using validation accuracy, never test outcomes."""
import argparse
import json
from pathlib import Path

import torch


def select_run(runs, output):
    candidates = []
    reference = None
    for directory in runs:
        directory = Path(directory)
        if not (directory / 'final.pt').exists():
            raise ValueError(f'Run is incomplete: {directory}')
        ckpt = torch.load(directory / 'best.pt', map_location='cpu', weights_only=True)
        if ckpt['full_data']:
            raise ValueError('Cannot select using a full-data run')
        config = ckpt['config']
        signature = (ckpt['archive_sha256'], config['split_seed'], config['val_fraction'])
        if reference is not None and reference != signature:
            raise ValueError('Runs must use identical data and validation splits')
        reference = signature
        candidates.append((ckpt['selection']['accuracy'], str(directory), ckpt))
    accuracy, directory, ckpt = max(candidates, key=lambda item: item[0])
    config = dict(ckpt['config'])
    config['inference_weights'] = ckpt['weights']
    # Preserve the full cosine schedule. Do not compress it to the best checkpoint epoch.
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(config, indent=2))
    report = {'selected_run': directory, 'validation_accuracy': accuracy,
              'best_epoch': ckpt['epoch'], 'final_fit_epochs': config['epochs'],
              'weights': ckpt['weights'], 'tta': config['tta'],
              'candidates': [{'run': d, 'accuracy': a} for a, d, _ in candidates]}
    Path(output).with_suffix('.selection.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    select_run(args.runs, args.out)
