"""不均衡対応の損失関数。設定ファイルの名前で切り替える。

欠陥ピクセルは全体の 0.1〜1% 程度しかない。素の BCE だと「全部背景」と答えるだけで
損失がほぼ下がるため、欠陥側を重視する工夫が必要になる。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SoftDiceLoss(nn.Module):
    """1 - Dice。予測と正解の「重なり」を直接最適化するので、背景の多さに引きずられにくい。

    バッチ全体でまとめて計算する。画像ごとに計算すると、良品 (正解が空) で 0/0 になるため。
    eps は分母ゼロ回避。
    """

    def __init__(self, eps: float = 1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        p = torch.sigmoid(logits)
        inter = (p * target).sum()
        return 1 - (2 * inter + self.eps) / (p.sum() + target.sum() + self.eps)


class FocalLoss(nn.Module):
    """簡単に当たる背景ピクセルの損失を (1-pt)^gamma で小さくし、難しい欠陥ピクセルに集中させる。"""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)  # 正解クラスに対する予測確率
        a = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (a * (1 - pt) ** self.gamma * bce).mean()


class Combo(nn.Module):
    """複数の損失を重み付きで足す (例: BCE + Dice)。"""

    def __init__(self, parts: list[tuple[float, nn.Module]]):
        super().__init__()
        self.weights = [w for w, _ in parts]
        self.parts = nn.ModuleList([m for _, m in parts])

    def forward(self, logits, target):
        return sum(w * m(logits, target) for w, m in zip(self.weights, self.parts))


class BCE(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        # pos_weight: 欠陥ピクセルの間違いを何倍重く罰するか (クラス重み)
        self.register_buffer("pw", torch.tensor(float(pos_weight)))

    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pw)


def build_loss(name: str, pos_weight: float = 1.0) -> nn.Module:
    if name == "bce":
        return BCE(pos_weight)
    if name == "dice":
        return SoftDiceLoss()
    if name == "bce_dice":
        return Combo([(1.0, BCE(pos_weight)), (1.0, SoftDiceLoss())])
    if name == "focal":
        return FocalLoss()
    if name == "focal_dice":
        return Combo([(1.0, FocalLoss()), (1.0, SoftDiceLoss())])
    raise ValueError(f"unknown loss: {name}")
