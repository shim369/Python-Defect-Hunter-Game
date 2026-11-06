import numpy as np
import pytest

from defect_hunter.data import resize_pair, stratified_split

LABELS = [1] * 25 + [0] * 200  # 実データと同程度の不均衡


def test_split_is_disjoint_and_complete():
    tr, va = stratified_split(LABELS, 0.2, seed=0)
    assert set(tr).isdisjoint(va)
    assert sorted(tr + va) == list(range(len(LABELS)))


def test_split_keeps_class_ratio():
    _, va = stratified_split(LABELS, 0.2, seed=0)
    assert sum(LABELS[i] for i in va) == 5  # 欠陥 25 枚の 20%
    assert len(va) == 45  # 全体 225 枚の 20%


def test_split_is_reproducible_and_seed_sensitive():
    assert stratified_split(LABELS, 0.2, 7) == stratified_split(LABELS, 0.2, 7)
    assert stratified_split(LABELS, 0.2, 7) != stratified_split(LABELS, 0.2, 8)


def test_split_rare_class_gets_at_least_one_val():
    _, va = stratified_split([1] + [0] * 50, 0.1, seed=0)
    assert sum(([1] + [0] * 50)[i] for i in va) == 1


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        stratified_split(LABELS, 1.0, 0)


def test_resize_pair_shapes_and_binary():
    img = np.random.default_rng(0).integers(0, 255, (640, 230, 3), dtype=np.uint8)
    mask = np.zeros((640, 230), np.uint8)
    mask[100:140, 50:90] = 1
    i2, m2 = resize_pair(img, mask, 320, 128)
    assert i2.shape == (320, 128, 3) and m2.shape == (320, 128)
    assert set(np.unique(m2)) <= {0, 1} and m2.any()
