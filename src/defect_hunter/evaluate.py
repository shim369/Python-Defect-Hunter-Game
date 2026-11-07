"""モデルの確率マップから、自前指標 (metrics.py) で IoU / P / R / F1 を計算する。

ピクセル単位: データセット全体の TP/FP/FN を合算してから指標を出す (micro 集計)。
  画像ごとの平均にすると、良品 (空マスク) の扱いで値が大きく変わるため。
画像単位: 「予測マスクの面積が min_area 以上なら欠陥あり」と判定し、混同行列を作る。
"""
from __future__ import annotations

import numpy as np
import torch

from . import metrics as M


@torch.no_grad()
def predict_probs(model, images_uint8: np.ndarray, batch_size: int = 32) -> np.ndarray:
    """(N,H,W,3) uint8 → (N,H,W) float32 の確率。BatchNorm を推論モードにするため eval()。"""
    model.eval()
    out = []
    for i in range(0, len(images_uint8), batch_size):
        x = torch.from_numpy(images_uint8[i : i + batch_size]).permute(0, 3, 1, 2).float() / 255.0
        out.append(torch.sigmoid(model(x))[:, 0].numpy())
    return np.concatenate(out)


def evaluate(probs: np.ndarray, masks: np.ndarray, threshold: float, min_area: int) -> dict:
    pred = probs >= threshold
    c = M.confusion(pred, masks)  # データセット全体で 1 つの混同行列 (micro)
    gt_img = masks.reshape(len(masks), -1).any(axis=1).tolist()
    pr_img = (pred.reshape(len(pred), -1).sum(axis=1) >= min_area).tolist()
    ci = M.image_level_confusion(pr_img, gt_img)
    return {
        "threshold": threshold,
        "pixel": {"iou": M.iou_from_counts(c), "precision": M.precision_from_counts(c),
                  "recall": M.recall_from_counts(c), "f1": M.f1_from_counts(c)},
        "image": {"precision": M.precision_from_counts(ci), "recall": M.recall_from_counts(ci),
                  "f1": M.f1_from_counts(ci), "tp": ci.tp, "fp": ci.fp, "fn": ci.fn, "tn": ci.tn},
    }


def best_threshold(probs, masks, min_area: int, grid=None) -> tuple[float, dict]:
    """val でピクセル F1 が最大になるしきい値を探す。test では絶対に探さない (漏れ防止)。"""
    # Dice 系の損失は確率が極端に低く出ることがあるため、下限を 0.01 まで細かく探す
    if grid is None:
        grid = np.unique(np.round(np.r_[0.01, 0.02, 0.03, np.arange(0.05, 0.96, 0.05), 0.97, 0.99], 2))
    results = [evaluate(probs, masks, float(t), min_area) for t in grid]
    best = max(results, key=lambda r: r["pixel"]["f1"])
    return best["threshold"], best
