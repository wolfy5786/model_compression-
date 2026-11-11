"""TEMPONet (Zanghieri et al., IEEE TBioCAS 2020) — FP32 baseline definition.

Architecture (paper):
  3 convolutional blocks; each has
    - 2 temporal Conv1d (k=3) with dilation d and same-length padding
    - 1 Conv1d (k=5) with stride s, raising channels
    - AvgPool1d(k=2, stride=2)
  Blocks use (d, s, channels) = (2,1,32), (4,2,64), (8,4,128).
  Then 2 FC layers with dropout, BN + ReLU throughout.
"""

from __future__ import annotations

import torch
from torch import nn


class _TempoBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int, stride: int):
        super().__init__()
        pad = dilation  # same length for k=3 dilated conv
        self.conv1 = nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.conv2 = nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(in_ch)
        # k=5 stride-s: pad so length shrinks mainly by stride (ceil)
        self.conv3 = nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2)
        self.bn3 = nn.BatchNorm1d(out_ch)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        return self.pool(x)


class TEMPONet(nn.Module):
    """Temporal Embedded Muscular Processing Online Network."""

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 50,
        dropout: float = 0.5,
        fc_hidden: int = 256,
        input_samples: int = 300,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.input_samples = input_samples

        self.block1 = _TempoBlock(in_channels, 32, dilation=2, stride=1)
        self.block2 = _TempoBlock(32, 64, dilation=4, stride=2)
        self.block3 = _TempoBlock(64, 128, dilation=8, stride=4)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_samples)
            flat = self._conv_forward(dummy).shape[1]

        self.fc1 = nn.Linear(flat, fc_hidden)
        self.bn_fc1 = nn.BatchNorm1d(fc_hidden)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
        self.act = nn.ReLU(inplace=True)

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, C, T) or (B, T, C)
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels and x.shape[2] == self.in_channels:
            x = x.transpose(1, 2)
        x = self._conv_forward(x)
        x = self.drop1(self.act(self.bn_fc1(self.fc1(x))))
        return self.fc2(x)
