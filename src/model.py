"""Pre-activation Wide ResNet, implemented from the paper (see REFERENCES.md)."""
import torch
from torch import nn
import torch.nn.functional as F


class WideBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.shortcut = (nn.Identity() if in_channels == out_channels and stride == 1
                         else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False))
        self.dropout = dropout

    def forward(self, x):
        pre = F.relu(self.bn1(x), inplace=False)
        skip = x if isinstance(self.shortcut, nn.Identity) else self.shortcut(pre)
        out = F.relu(self.bn2(self.conv1(pre)), inplace=True)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return skip + self.conv2(out)


class WideResNet(nn.Module):
    def __init__(self, depth=28, base_width=56, dropout=0.0, num_classes=10):
        super().__init__()
        if depth < 10 or (depth - 4) % 6:
            raise ValueError('depth must be 6n+4, n >= 1')
        blocks = (depth - 4) // 6
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        stages, previous = [], 16
        for stage, width in enumerate([base_width, base_width * 2, base_width * 4]):
            for block in range(blocks):
                stride = 2 if stage > 0 and block == 0 else 1
                stages.append(WideBlock(previous, width, stride, dropout))
                previous = width
        self.blocks = nn.Sequential(*stages)
        self.bn = nn.BatchNorm2d(previous)
        self.head = nn.Linear(previous, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.blocks(self.stem(x))), inplace=True)
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


def build_model(config):
    model = WideResNet(**config['model'])
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count > 5_000_000:
        raise ValueError(f'Competition limit exceeded: {count:,} > 5,000,000')
    return model
