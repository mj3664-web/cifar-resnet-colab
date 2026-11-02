"""CPU-only integration test on synthetic fixtures, never competition training."""
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.data import prepare


class EndToEndTest(unittest.TestCase):
    def test_train_resume_select_full_fit_and_predict(self):
        self.run_recipe('main')

    def test_augmix_train_resume_select_full_fit_and_predict(self):
        self.run_recipe('wide_augmix')

    def run_recipe(self, recipe):
        with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='.test-') as temp:
            root = Path(temp)
            rng = np.random.default_rng(1)
            archive = root / 'fixture.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                train_rows = ['id,filename,label']
                for i in range(60):
                    image = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
                    buf = io.BytesIO()
                    image.save(buf, format='PNG')
                    z.writestr(f'train_images/train_{i}.png', buf.getvalue())
                    train_rows.append(f'{i},train_{i}.png,{i % 10}')
                z.writestr('train_labels.csv', '\r\r\n'.join(train_rows))
                test_rows = ['id,filename']
                for i in range(3):
                    buf = io.BytesIO()
                    Image.new('RGB', (32, 32), (i * 50, 80, 100)).save(buf, format='PNG')
                    z.writestr(f'test_images/test_{i}.png', buf.getvalue())
                    test_rows.append(f'00{i},test_{i}.png')
                z.writestr('test.csv', '\n'.join(test_rows))
                z.writestr('sample_submission.csv', 'id,label\n002,0\n000,0\n001,0\n')
            cache = root / 'cache'
            prepare(archive, cache)
            config = json.loads(Path(f'configs/{recipe}.json').read_text())
            config.update(model={'depth': 10, 'base_width': 4, 'dropout': 0.1}, epochs=2,
                          warmup_epochs=0, batch_size=16, mix_off_last=0)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}

            def run(module, *args):
                result = subprocess.run([sys.executable, '-m', module, *map(str, args)],
                                        capture_output=True, text=True, env=env)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            common = ['--data', cache, '--config', config_path, '--workers', 0,
                      '--device', 'cpu', '--deterministic']
            uninterrupted, resumed = root / 'continuous', root / 'resumed'
            run('src.train', *common, '--out', uninterrupted)
            run('src.train', *common, '--out', resumed, '--stop-after', 1)
            run('src.train', *common, '--out', resumed, '--resume')
            a = torch.load(uninterrupted / 'last.pt', weights_only=False, map_location='cpu')
            b = torch.load(resumed / 'last.pt', weights_only=False, map_location='cpu')
            for group in ['model', 'ema']:
                for name in a[group]:
                    torch.testing.assert_close(a[group][name], b[group][name], rtol=0, atol=0)
            self.assertEqual(a['epoch'], 2)
            self.assertEqual(a['best_info'], b['best_info'])
            selected = root / 'selected.json'
            run('src.select', '--runs', uninterrupted, resumed, '--out', selected)
            run('src.evaluate', '--data', cache, '--checkpoint', resumed / 'best.pt', '--workers', 0)
            final = root / 'final'
            run('src.train', '--data', cache, '--config', selected, '--out', final,
                '--full-data', '--workers', 0, '--device', 'cpu', '--deterministic')
            run('src.predict', '--data', cache, '--checkpoint', final / 'final.pt',
                '--out', root / 'submission.csv', '--workers', 0)
            with (root / 'submission.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['id'] for r in rows], ['002', '000', '001'])
            self.assertTrue(all(0 <= int(r['label']) <= 9 for r in rows))
            meta = json.loads((root / 'submission.metadata.json').read_text())
            self.assertTrue(meta['full_data'])
            self.assertEqual(meta['rows'], 3)
            if recipe == 'wide_augmix':
                history = [json.loads(s) for s in (resumed / 'history.jsonl').read_text().splitlines()]
                self.assertTrue(all(r['jsd'] >= -1e-6 and r['clean_ce'] > 0 for r in history))
                run('src.robustness', '--data', cache, '--checkpoints', resumed / 'best.pt',
                    '--out', root / 'robustness.json', '--workers', 0)
                result = json.loads((root / 'robustness.json').read_text())
                self.assertEqual(len(result['results'][0]['accuracy']), 5)


if __name__ == '__main__':
    unittest.main()
