"""小型 U-Net。CPU でも回せるよう、チャンネル数を小さく (base=16) している。"""
from __future__ import annotations

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    # 3x3 畳み込みを 2 回。BatchNorm は学習の安定化、ReLU は非線形性のため。
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """エンコーダで縮小して文脈を取り、デコーダで拡大しつつスキップ接続で細部を戻す。

    欠陥は小さいので、スキップ接続 (縮小前の高解像度特徴を渡す) が重要になる。
    入力の高さ・幅は 2**depth の倍数にすること (320x128 は depth=4 で OK)。
    出力は「ロジット」(sigmoid 前の値)。sigmoid は損失側・推論側で掛ける。
    """

    def __init__(self, in_ch: int = 3, base: int = 16, depth: int = 4):
        super().__init__()
        chs = [base * 2**i for i in range(depth + 1)]
        self.down = nn.ModuleList([_block(in_ch, chs[0])] + [_block(chs[i], chs[i + 1]) for i in range(depth)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(chs[i + 1], chs[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([_block(chs[i] * 2, chs[i]) for i in reversed(range(depth))])
        self.head = nn.Conv2d(chs[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, blk in enumerate(self.down):
            x = blk(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips.pop()  # 最深部
        for up, dec in zip(self.up, self.dec):
            x = up(x)
            x = dec(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)
