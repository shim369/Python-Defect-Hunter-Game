"""自前実装の評価指標 (IoU / Precision / Recall / F1)。

外部ライブラリ(sklearn, torchmetrics 等)は使わず、numpy だけで書く。
tests/test_metrics_vs_torchmetrics.py で torchmetrics と結果を突き合わせる。

どの指標も「分母が 0」になる場合がある (例: 正解も予測も空マスク)。
その扱いは引数で明示させ、黙って 0 や NaN を返さないようにする。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Confusion:
    """混同行列の 4 つのカウント。ピクセル単位でも画像単位でも使える。"""

    tp: int
    fp: int
    fn: int
    tn: int


def _to_bool(x: np.ndarray) -> np.ndarray:
    # 0/1 でも 0/255 でも bool でも受けられるよう、「0 より大きい=欠陥」に統一する。
    # 確率マップを渡す場合は、呼び出し側で先にしきい値処理すること。
    return np.asarray(x) > 0


def confusion(pred: np.ndarray, gt: np.ndarray) -> Confusion:
    """2 つの 2 値マスクから TP/FP/FN/TN を数える。"""
    p, g = _to_bool(pred), _to_bool(gt)
    if p.shape != g.shape:
        # 黙ってブロードキャストされると誤った点数になるため、ここで止める。
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")
    return Confusion(
        tp=int(np.sum(p & g)),
        fp=int(np.sum(p & ~g)),
        fn=int(np.sum(~p & g)),
        tn=int(np.sum(~p & ~g)),
    )


def _ratio(num: int, den: int, zero_division: float) -> float:
    return zero_division if den == 0 else num / den


def iou_from_counts(c: Confusion, empty_value: float = 1.0) -> float:
    """IoU = TP / (TP + FP + FN)。和集合が空 (両方空マスク) のときは empty_value。

    デフォルト 1.0: 「欠陥なし」と正しく答えたら満点、というゲーム規則に合わせた。
    """
    return _ratio(c.tp, c.tp + c.fp + c.fn, empty_value)


def precision_from_counts(c: Confusion, zero_division: float = 0.0) -> float:
    """Precision = TP / (TP + FP)。何も予測しなかった場合は zero_division。"""
    return _ratio(c.tp, c.tp + c.fp, zero_division)


def recall_from_counts(c: Confusion, zero_division: float = 0.0) -> float:
    """Recall = TP / (TP + FN)。正解が空の場合は zero_division。"""
    return _ratio(c.tp, c.tp + c.fn, zero_division)


def f1_from_counts(c: Confusion, zero_division: float = 0.0) -> float:
    """F1 = 2TP / (2TP + FP + FN)。P と R の調和平均と等価で、0 除算が起きにくい形。"""
    return _ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn, zero_division)


# --- マスク同士を直接渡す版 (ピクセル単位) ---


def iou(pred: np.ndarray, gt: np.ndarray, empty_value: float = 1.0) -> float:
    return iou_from_counts(confusion(pred, gt), empty_value)


def precision(pred: np.ndarray, gt: np.ndarray, zero_division: float = 0.0) -> float:
    return precision_from_counts(confusion(pred, gt), zero_division)


def recall(pred: np.ndarray, gt: np.ndarray, zero_division: float = 0.0) -> float:
    return recall_from_counts(confusion(pred, gt), zero_division)


def f1(pred: np.ndarray, gt: np.ndarray, zero_division: float = 0.0) -> float:
    return f1_from_counts(confusion(pred, gt), zero_division)


def image_level_confusion(pred_has_defect: list[bool], gt_has_defect: list[bool]) -> Confusion:
    """画像単位の判定 (欠陥あり/なし) から混同行列を作る。ゲームの P/R/F1 表示で使う。"""
    if len(pred_has_defect) != len(gt_has_defect):
        raise ValueError("length mismatch")
    p = np.asarray(pred_has_defect, dtype=bool)
    g = np.asarray(gt_has_defect, dtype=bool)
    return confusion(p, g)
