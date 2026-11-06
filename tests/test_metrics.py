import numpy as np
import pytest

from defect_hunter import metrics as m


def mask(rows):
    return np.array(rows, dtype=np.uint8)


EMPTY = np.zeros((4, 4), dtype=np.uint8)
FULL = np.ones((4, 4), dtype=np.uint8)


def test_perfect_match():
    g = mask([[1, 1, 0], [0, 1, 0]])
    assert m.iou(g, g) == 1.0
    assert m.precision(g, g) == 1.0
    assert m.recall(g, g) == 1.0
    assert m.f1(g, g) == 1.0


def test_complete_mismatch():
    p = mask([[1, 1, 0, 0]])
    g = mask([[0, 0, 1, 1]])
    assert m.iou(p, g) == 0.0
    assert m.precision(p, g) == 0.0
    assert m.recall(p, g) == 0.0
    assert m.f1(p, g) == 0.0


def test_both_empty_uses_explicit_conventions():
    assert m.iou(EMPTY, EMPTY) == 1.0  # デフォルト: 「欠陥なし」正解は満点
    assert m.iou(EMPTY, EMPTY, empty_value=0.0) == 0.0
    assert m.precision(EMPTY, EMPTY) == 0.0
    assert m.precision(EMPTY, EMPTY, zero_division=1.0) == 1.0
    assert m.recall(EMPTY, EMPTY, zero_division=1.0) == 1.0
    assert m.f1(EMPTY, EMPTY, zero_division=1.0) == 1.0


def test_pred_empty_gt_defect():
    # 見逃し: 何も予測せず、正解には欠陥がある
    assert m.iou(EMPTY, FULL) == 0.0
    assert m.recall(EMPTY, FULL) == 0.0
    assert m.precision(EMPTY, FULL) == 0.0  # 予測ゼロ -> zero_division
    assert m.precision(EMPTY, FULL, zero_division=1.0) == 1.0
    assert m.f1(EMPTY, FULL) == 0.0


def test_pred_defect_gt_empty():
    # 誤検出: 正解は良品なのに塗った
    assert m.iou(FULL, EMPTY) == 0.0
    assert m.precision(FULL, EMPTY) == 0.0
    assert m.recall(FULL, EMPTY) == 0.0  # 正解ゼロ -> zero_division
    assert m.recall(FULL, EMPTY, zero_division=1.0) == 1.0


def test_partial_overlap_hand_computed():
    p = mask([[1, 1, 1, 0]])  # 予測 3 px
    g = mask([[0, 1, 1, 1]])  # 正解 3 px, 重なり 2 px
    c = m.confusion(p, g)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 0)
    assert m.iou(p, g) == pytest.approx(2 / 4)
    assert m.precision(p, g) == pytest.approx(2 / 3)
    assert m.recall(p, g) == pytest.approx(2 / 3)
    assert m.f1(p, g) == pytest.approx(2 / 3)


def test_f1_is_harmonic_mean_of_p_and_r():
    p = mask([[1, 1, 1, 1, 0, 0]])
    g = mask([[0, 1, 1, 0, 0, 1]])
    P, R = m.precision(p, g), m.recall(p, g)
    assert m.f1(p, g) == pytest.approx(2 * P * R / (P + R))


def test_accepts_0_255_and_bool():
    g = mask([[255, 0, 255, 0]])
    p = np.array([[True, False, False, False]])
    assert m.iou(p, g) == pytest.approx(1 / 2)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        m.iou(np.zeros((2, 2)), np.zeros((3, 3)))


def test_image_level_confusion():
    c = m.image_level_confusion([True, True, False, False], [True, False, True, False])
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert m.precision_from_counts(c) == 0.5
    assert m.recall_from_counts(c) == 0.5
    assert m.f1_from_counts(c) == 0.5
    with pytest.raises(ValueError):
        m.image_level_confusion([True], [True, False])
