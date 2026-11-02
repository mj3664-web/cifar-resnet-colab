import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data import stratified_split
from src.engine import EMA, learning_rate, mix_batch
from src.model import build_model
from src.predict import write_submission


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads(Path('configs/main.json').read_text())

    def test_exact_model_size_and_gradient(self):
        model = build_model(self.config)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4_480_474)
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 10))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 9])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        oversized = {**self.config, 'model': {'depth': 28, 'base_width': 64}}
        with self.assertRaises(ValueError):
            build_model(oversized)

    def test_split_disjoint_and_repeatable(self):
        labels = np.tile(np.arange(10), 100)
        train, val = stratified_split(labels)
        other_train, other_val = stratified_split(labels)
        self.assertEqual(len(set(train) & set(val)), 0)
        self.assertEqual(len(set(train) | set(val)), len(labels))
        np.testing.assert_array_equal(np.bincount(labels[val]), np.full(10, 10))
        np.testing.assert_array_equal(train, other_train)
        np.testing.assert_array_equal(val, other_val)

    def test_submission_uses_ids_not_row_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sample = tmp / 'sample.csv'
            sample.write_text('id,label\n001,0\n9,0\nabc,0\n')
            rows = [{'id': 'abc'}, {'id': '001'}, {'id': '9'}]
            out = tmp / 'submission.csv'
            self.assertEqual(write_submission(sample, rows, [7, 2, 0], out), 3)
            with out.open(newline='') as stream:
                self.assertEqual(list(csv.reader(stream)), [['id', 'label'], ['001', '2'], ['9', '0'], ['abc', '7']])
            with self.assertRaises(ValueError):
                write_submission(sample, rows, [10, 2, 0], out)
            with self.assertRaises(ValueError):
                write_submission(sample, [rows[0]] * 3, [1, 2, 3], out)

    def test_cutmix_area_matches_pixels(self):
        config = {**self.config, 'mix_probability': 1, 'mixup_alpha': 0, 'cutmix_alpha': 1}
        x = torch.arange(8).float().reshape(8, 1, 1, 1).expand(8, 3, 32, 32).clone()
        y = torch.arange(8)
        for seed in range(10):
            np.random.seed(seed)
            torch.manual_seed(seed)
            mixed, ya, yb, lam = mix_batch(x, y, config)
            expected = ya.float() * lam + yb.float() * (1 - lam)
            torch.testing.assert_close(mixed.mean((1, 2, 3)), expected)

    def test_ema_has_no_trainable_parameters(self):
        model = build_model({'model': {'depth': 10, 'base_width': 4}})
        ema = EMA(model, 0.999)
        self.assertFalse(any(p.requires_grad for p in ema.model.parameters()))
        with torch.no_grad():
            next(model.parameters()).add_(1)
        ema.update(model)
        self.assertEqual(ema.updates, 1)
        self.assertFalse(torch.equal(next(model.parameters()), next(ema.model.parameters())))

    def test_cosine_endpoints(self):
        c = self.config
        self.assertAlmostEqual(learning_rate(c, c['warmup_epochs']), c['lr'])
        self.assertAlmostEqual(learning_rate(c, c['epochs']), c['min_lr'])


if __name__ == '__main__':
    unittest.main()
