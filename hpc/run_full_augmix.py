"""Run only the full-data wide AugMix fit inside a scheduled GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torchvision
from src.data import digest
from src.model import build_model

EXPECTED_ARCHIVE = '397e06e5752bfe0576d1258765bbe1d020cca02db2fc21442cbab6d50257ca51'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True, help='Persistent project directory on scratch / project storage')
    p.add_argument('--data-zip', help='Defaults to PROJECT/data.zip')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--validate-only', action='store_true', help='Validate code/config on CPU; do not train or submit')
    args = p.parse_args()
    config_path = REPO / 'configs/wide_augmix.json'
    config = json.loads(config_path.read_text())
    model = build_model(config)
    count = sum(v.numel() for v in model.parameters() if v.requires_grad)
    if count != 4_972_345 or config['epochs'] != 400 or config['augmentation'] != 'augmix':
        raise ValueError('Expected the agreed 400-epoch, 4,972,345-parameter AugMix configuration')
    print(json.dumps({'parameter_count': count, 'epochs': config['epochs'],
                      'batch_size': config['batch_size'], 'weights': config['inference_weights'],
                      'torch': torch.__version__, 'torchvision': torchvision.__version__}), flush=True)
    del model
    if args.validate_only:
        print('Code/config validation passed. No training or job submission performed.')
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Launch through sbatch; do not train on a login node')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use a CUDA-enabled torch/torchvision environment and a GPU allocation.')
    project = Path(args.project).expanduser().resolve()
    project.mkdir(parents=True, exist_ok=True)
    # Lock is held by this process while the training/prediction subprocess runs.
    # The kernel releases it on interruption; no manual stale-lock removal is required.
    import fcntl
    with (project / '.full_augmix.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another full-data AugMix job already holds this project lock')
        archive = Path(args.data_zip).expanduser().resolve() if args.data_zip else project / 'data.zip'
        if not archive.is_file():
            raise FileNotFoundError(f'Upload the original archive to {archive}')
        if digest(archive) != EXPECTED_ARCHIVE:
            raise ValueError('Dataset ZIP hash differs from the original supplied competition archive')
        cache, run = project / 'cache', project / 'runs/full_wide_augmix'
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
        print('Project:', project, flush=True)
        print('Checkpoint directory:', run, flush=True)

        def execute(module, *values):
            command = [sys.executable, '-u', '-m', module, *map(str, values)]
            print('Running:', ' '.join(command), flush=True)
            subprocess.run(command, cwd=REPO, check=True)

        # Rebuild cache on each new job. This is inexpensive for this archive and avoids
        # using an interrupted or stale cache when a previous Slurm allocation ended.
        execute('src.data', '--zip', archive, '--out', cache)
        manifest = json.loads((cache / 'manifest.json').read_text())
        if manifest['train_count'] != 50000 or manifest['test_count'] != 2000:
            raise ValueError('Unexpected competition dataset sizes')
        run.mkdir(parents=True, exist_ok=True)
        provenance = {'slurm_job_id': os.environ['SLURM_JOB_ID'],
                      'slurm_account': os.environ.get('SLURM_JOB_ACCOUNT'),
                      'code_directory': str(REPO), 'gpu': torch.cuda.get_device_name(0),
                      'decision': 'User chose full-data AugMix before development selection completed.',
                      'validation_accuracy': None}
        pin = REPO / 'code_commit.txt'
        if pin.exists():
            provenance['code_commit'] = pin.read_text().strip()
        (run / f'job_{os.environ["SLURM_JOB_ID"]}.json').write_text(json.dumps(provenance, indent=2))
        execute('src.train', '--data', cache, '--config', config_path, '--out', run,
                '--full-data', '--resume', '--workers', args.workers, '--device', 'cuda')
        checkpoint = run / 'final.pt'
        if not checkpoint.is_file():
            raise RuntimeError('Full-data training has not produced final.pt')
        execute('src.predict', '--data', cache, '--checkpoint', checkpoint,
                '--out', project / 'submission/submission.csv', '--workers', args.workers)
        print('Completed. Save the final checkpoint and submission outside scratch for long-term retention.', flush=True)


if __name__ == '__main__':
    main()
