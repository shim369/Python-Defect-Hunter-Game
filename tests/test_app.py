import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from defect_hunter.app import create_app

H, W = 32, 16


def _png_b64(mask: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", (mask * 255).astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


@pytest.fixture()
def client(tmp_path):
    """欠陥あり 3 枚 (d0-d2)、良品 3 枚 (g0-g2) の合成素材。AI は d0 だけ完全一致、g0 を誤検出、d1 を見逃す。"""
    items = []
    box = np.zeros((H, W), np.uint8)
    box[4:12, 4:12] = 1
    for name, defect in [("d0", 1), ("d1", 1), ("d2", 1), ("g0", 0), ("g1", 0), ("g2", 0)]:
        gt = box if defect else np.zeros((H, W), np.uint8)
        prob = {"d0": box, "g0": box}.get(name, np.zeros((H, W), np.uint8))
        cv2.imwrite(str(tmp_path / f"{name}.jpg"), np.full((H, W, 3), 128, np.uint8))
        cv2.imwrite(str(tmp_path / f"{name}_gt.png"), gt * 255)
        cv2.imwrite(str(tmp_path / f"{name}_prob.png"), prob * 200)  # 確率 約0.78
        items.append({"id": name, "has_defect": defect})
    (tmp_path / "index.json").write_text(json.dumps(
        {"model": "t", "threshold": 0.5, "min_area": 4, "height": H, "width": W, "items": items}))
    return TestClient(create_app(tmp_path))


def test_round_ratio_and_no_label_leak(client):
    r = client.get("/api/round?n=10&defect_ratio=0.5&seed=1").json()["questions"]
    assert len(r) == 6  # 素材が少ないので 3+3 に頭打ち
    assert all(set(q) == {"id", "image_url"} for q in r)  # 欠陥の有無を返さない
    assert r == client.get("/api/round?n=10&defect_ratio=0.5&seed=1").json()["questions"]  # seed 固定で再現


def test_round_rejects_bad_params(client):
    assert client.get("/api/round?n=1").status_code == 422


def test_perfect_answer_scores_one(client):
    box = np.zeros((H, W), np.uint8)
    box[4:12, 4:12] = 1
    r = client.post("/api/answer", json={"id": "d0", "mask": _png_b64(box)}).json()
    assert r["player_iou"] == 1.0 and r["ai_iou"] == 1.0
    assert r["gt_has_defect"] and r["player_has_defect"] and r["ai_has_defect"]


def test_no_defect_on_good_is_full_marks_and_on_defect_is_zero(client):
    good = client.post("/api/answer", json={"id": "g1", "no_defect": True}).json()
    assert good["player_iou"] == 1.0 and not good["gt_has_defect"]
    miss = client.post("/api/answer", json={"id": "d1", "no_defect": True}).json()
    assert miss["player_iou"] == 0.0 and miss["ai_iou"] == 0.0  # AI も d1 は見逃し


def test_painting_on_good_scores_zero(client):
    m = np.zeros((H, W), np.uint8)
    m[0:6, 0:6] = 1
    r = client.post("/api/answer", json={"id": "g1", "mask": _png_b64(m)}).json()
    assert r["player_iou"] == 0.0 and r["player_has_defect"]
    fp = client.post("/api/answer", json={"id": "g0", "no_defect": True}).json()
    assert fp["ai_iou"] == 0.0 and fp["ai_has_defect"]  # AI の誤検出


def test_bad_inputs(client):
    assert client.post("/api/answer", json={"id": "nope", "no_defect": True}).status_code == 404
    assert client.post("/api/answer", json={"id": "../index", "no_defect": True}).status_code == 404
    wrong = np.zeros((5, 5), np.uint8)
    assert client.post("/api/answer", json={"id": "d0", "mask": _png_b64(wrong)}).status_code == 422
    assert client.post("/api/answer", json={"id": "d0", "mask": "not-a-png"}).status_code == 422


def test_summary_computes_prf_from_server_truth(client):
    # d0: 当てた, d1: 見逃し, g0: 誤検出(プレイヤー), g1: 正しく良品
    ans = [{"id": "d0", "player_has_defect": True}, {"id": "d1", "player_has_defect": False},
           {"id": "g0", "player_has_defect": True}, {"id": "g1", "player_has_defect": False}]
    s = client.post("/api/summary", json={"answers": ans}).json()
    p = s["player"]
    assert (p["tp"], p["fp"], p["fn"], p["tn"]) == (1, 1, 1, 1)
    assert p["precision"] == p["recall"] == p["f1"] == 0.5
    a = s["ai"]  # AI: d0 当たり, d1 見逃し, g0 誤検出, g1 正解 -> 同じ混同行列
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (1, 1, 1, 1)


def test_summary_undefined_metrics_are_null_and_threshold_matters(client):
    s = client.post("/api/summary", json={"answers": [{"id": "g1", "player_has_defect": False}]}).json()
    assert s["player"]["precision"] is None and s["player"]["recall"] is None  # 分母 0 -> null
    both = [{"id": "d0", "player_has_defect": True}, {"id": "g0", "player_has_defect": True}]
    lo = client.post("/api/summary", json={"answers": both, "threshold": 0.5}).json()["ai"]
    hi = client.post("/api/summary", json={"answers": both, "threshold": 0.9}).json()["ai"]
    assert lo["fp"] == 1 and hi["fp"] == 0 and hi["fn"] == 1  # しきい値 0.9 (> 0.78) だと何も検出しない


def test_static_pages_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/game.js").status_code == 200
    assert client.get("/assets/d0.jpg").status_code == 200
    assert client.get("/api/config").json()["width"] == W
