import numpy as np
import torch

from defect_hunter.evaluate import best_threshold, evaluate
from defect_hunter.losses import FocalLoss, SoftDiceLoss, build_loss
from defect_hunter.model import UNet


def test_unet_output_shape_matches_input():
    y = UNet(base=4)(torch.randn(2, 3, 64, 32))
    assert y.shape == (2, 1, 64, 32)


def test_dice_zero_on_perfect_and_high_on_wrong():
    t = torch.zeros(1, 1, 8, 8)
    t[..., :2, :2] = 1
    perfect = (t * 2 - 1) * 20  # 十分大きいロジットで ほぼ one-hot
    assert SoftDiceLoss()(perfect, t).item() < 0.05
    assert SoftDiceLoss()(-perfect, t).item() > 0.8


def test_dice_handles_all_empty_batch():
    # 良品だけのバッチでも NaN にならない
    v = SoftDiceLoss()(torch.full((2, 1, 8, 8), -10.0), torch.zeros(2, 1, 8, 8))
    assert torch.isfinite(v)


def test_focal_downweights_easy_pixels_vs_bce():
    t = torch.zeros(1, 1, 4, 4)
    easy = torch.full((1, 1, 4, 4), -5.0)
    bce = build_loss("bce")(easy, t)
    assert FocalLoss()(easy, t) < bce


def test_pos_weight_increases_loss_on_missed_defect():
    t = torch.ones(1, 1, 4, 4)
    miss = torch.full((1, 1, 4, 4), -3.0)
    assert build_loss("bce", 10)(miss, t) > build_loss("bce", 1)(miss, t)


def test_evaluate_perfect_prediction():
    masks = np.zeros((4, 16, 16), np.uint8)
    masks[0, 2:6, 2:6] = 1
    r = evaluate(masks.astype(np.float32), masks, 0.5, min_area=4)
    assert r["pixel"]["iou"] == 1.0 and r["pixel"]["f1"] == 1.0
    assert r["image"]["tp"] == 1 and r["image"]["tn"] == 3
    thr, _ = best_threshold(masks.astype(np.float32), masks, 4)
    assert 0 < thr < 1
