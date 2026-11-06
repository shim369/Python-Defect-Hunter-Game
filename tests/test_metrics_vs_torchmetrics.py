"""自前指標が torchmetrics (公式実装) と一致することを、ランダムなマスクで確認する。"""
import numpy as np
import pytest
import torch
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from defect_hunter import metrics as m


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("density", [0.02, 0.3, 0.8])
def test_matches_torchmetrics(seed, density):
    rng = np.random.default_rng(seed)  # 乱数シード固定で再現可能にする
    p = (rng.random((32, 32)) < density).astype(np.uint8)
    g = (rng.random((32, 32)) < density).astype(np.uint8)
    tp, tg = torch.from_numpy(p), torch.from_numpy(g)

    assert m.iou(p, g) == pytest.approx(BinaryJaccardIndex()(tp, tg).item(), abs=1e-6)
    assert m.precision(p, g) == pytest.approx(BinaryPrecision()(tp, tg).item(), abs=1e-6)
    assert m.recall(p, g) == pytest.approx(BinaryRecall()(tp, tg).item(), abs=1e-6)
    assert m.f1(p, g) == pytest.approx(BinaryF1Score()(tp, tg).item(), abs=1e-6)
