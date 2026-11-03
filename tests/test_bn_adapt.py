import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.bn_adapt import recalibrate


class BNAdaptTests(unittest.TestCase):
    def test_only_bn_statistics_change(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.Dropout(0.9),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))
        original = {k: v.clone() for k, v in model.state_dict().items()}
        modes = []
        handle = model.register_forward_pre_hook(lambda m, x: modes.append((m.training, m[1].training, m[2].training)))
        ds = TensorDataset(torch.randn(20, 3, 8, 8) + 2, torch.arange(20))
        report = recalibrate(model, DataLoader(ds, batch_size=5), torch.device('cpu'))
        handle.remove()
        self.assertEqual(report['images'], 20)
        self.assertEqual(report['batches'], 4)
        self.assertTrue(all(x == (False, True, False) for x in modes))
        self.assertFalse(any(m.training for m in model.modules()))
        self.assertEqual(model[1].num_batches_tracked.item(), 4)
        self.assertEqual(model[1].momentum, 0.1)
        self.assertFalse(torch.equal(original['1.running_mean'], model[1].running_mean))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
            self.assertIsNone(parameter.grad)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            model(ds.tensors[0])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
