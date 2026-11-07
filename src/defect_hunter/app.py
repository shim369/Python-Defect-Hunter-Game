"""ゲームの API (FastAPI)。

方針: サーバーでモデルは動かさない。事前生成した「確率マップ PNG」を読み、
しきい値処理・採点 (自前指標 metrics.py) だけを行う。正解はサーバー側だけが持ち、
クライアントが送るのは「なぞったマスク」のみ (正解の改ざんを防ぐ)。
"""
from __future__ import annotations

import base64
import json
import os
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import metrics as M

WEB_DIR = Path(__file__).parent / "web" / "static"
MAX_MASK_B64 = 400_000  # 異常に大きい入力を弾く


class AnswerIn(BaseModel):
    id: str
    no_defect: bool = False
    mask: str | None = None  # 128x320 の PNG (data URL でも base64 でも可)。白 = 欠陥


class SummaryItem(BaseModel):
    id: str
    player_has_defect: bool


class SummaryIn(BaseModel):
    answers: list[SummaryItem] = Field(min_length=1, max_length=100)
    threshold: float | None = Field(default=None, ge=0, le=1)


def _undefined_to_none(v: float) -> float | None:
    # 分母 0 (例: 一度も「欠陥あり」と答えなかった → Precision 未定義) は None にして画面で「—」表示
    return None if v < 0 else v


def _prf(c: M.Confusion) -> dict:
    return {
        "precision": _undefined_to_none(M.precision_from_counts(c, zero_division=-1)),
        "recall": _undefined_to_none(M.recall_from_counts(c, zero_division=-1)),
        "f1": _undefined_to_none(M.f1_from_counts(c, zero_division=-1)),
        "tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn,
    }


def create_app(assets_dir: str | Path | None = None) -> FastAPI:
    assets = Path(assets_dir or os.environ.get("GAME_ASSETS_DIR", "outputs/game"))
    index_path = assets / "index.json"
    if not index_path.exists():
        raise RuntimeError(f"{index_path} がありません。`make assets` で素材を生成してください。")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    has_defect = {it["id"]: bool(it["has_defect"]) for it in index["items"]}
    pos_ids = sorted(i for i, v in has_defect.items() if v)
    neg_ids = sorted(i for i, v in has_defect.items() if not v)
    default_thr, min_area = index["threshold"], index["min_area"]
    shape = (index["height"], index["width"])

    @lru_cache(maxsize=2048)
    def read_png(name: str) -> np.ndarray:
        img = cv2.imread(str(assets / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise HTTPException(500, f"asset missing: {name}")
        return img

    def gt_mask(id_: str) -> np.ndarray:
        return read_png(f"{id_}_gt.png") > 0

    def ai_mask(id_: str, thr: float) -> np.ndarray:
        # 確率マップ (0-255) をしきい値処理する。ブラウザのスライダーと同じ計算式。
        return read_png(f"{id_}_prob.png").astype(np.float32) / 255.0 >= thr

    def check_id(id_: str) -> None:
        if id_ not in has_defect:  # 既知 ID のみ許可 (パス操作の防止も兼ねる)
            raise HTTPException(404, "unknown id")

    app = FastAPI(title="Python-Defect-Hunter-Game")

    @app.get("/api/config")
    def config():
        return {"model": index["model"], "default_threshold": default_thr, "min_area": min_area,
                "height": shape[0], "width": shape[1]}

    @app.get("/api/round")
    def new_round(n: int = 10, defect_ratio: float = 0.6, seed: int | None = None):
        """欠陥あり:良品 = 6:4 で出題 (実データは約 1:9 だとほぼ欠陥が出ず遊べないため)。"""
        if not 2 <= n <= 50 or not 0 <= defect_ratio <= 1:
            raise HTTPException(422, "bad parameters")
        rng = random.Random(seed)
        n_pos = min(round(n * defect_ratio), len(pos_ids))
        ids = rng.sample(pos_ids, n_pos) + rng.sample(neg_ids, min(n - n_pos, len(neg_ids)))
        rng.shuffle(ids)
        # 欠陥の有無が分かる情報は返さない
        return {"questions": [{"id": i, "image_url": f"/assets/{i}.jpg"} for i in ids]}

    @app.post("/api/answer")
    def answer(a: AnswerIn):
        check_id(a.id)
        gt = gt_mask(a.id)
        if a.no_defect or not a.mask:
            player = np.zeros(shape, bool)
        else:
            if len(a.mask) > MAX_MASK_B64:
                raise HTTPException(413, "mask too large")
            try:
                raw = base64.b64decode(a.mask.split(",")[-1], validate=False)
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            except Exception:
                img = None
            if img is None or img.shape != shape:
                raise HTTPException(422, f"mask must be a PNG of {shape[1]}x{shape[0]}")
            player = img > 127
        ai = ai_mask(a.id, default_thr)
        # 空マスクの IoU は 1.0 (良品に「欠陥なし」なら満点)。metrics.iou の既定値と同じ。
        return {
            "id": a.id,
            "gt_has_defect": bool(gt.any()),
            "player_iou": M.iou(player, gt),
            "ai_iou": M.iou(ai, gt),
            "player_has_defect": bool(player.sum() >= min_area),
            "ai_has_defect": bool(ai.sum() >= min_area),
            "gt_url": f"/assets/{a.id}_gt.png",
            "prob_url": f"/assets/{a.id}_prob.png",
            "default_threshold": default_thr,
        }

    @app.post("/api/summary")
    def summary(s: SummaryIn):
        """ここまでの答えから、プレイヤーと AI の画像単位 P/R/F1 を計算する。

        threshold を動かすと AI の判定が変わり、Precision/Recall のトレードオフが見える。
        """
        thr = default_thr if s.threshold is None else s.threshold
        for x in s.answers:
            check_id(x.id)
        gt = [has_defect[x.id] for x in s.answers]
        player = [x.player_has_defect for x in s.answers]
        ai_masks = [ai_mask(x.id, thr) for x in s.answers]
        ai = [bool(m.sum() >= min_area) for m in ai_masks]
        ai_iou = float(np.mean([M.iou(m, gt_mask(x.id)) for m, x in zip(ai_masks, s.answers)]))
        return {
            "n": len(s.answers), "threshold": thr,
            "player": _prf(M.image_level_confusion(player, gt)),
            "ai": {**_prf(M.image_level_confusion(ai, gt)), "mean_iou": ai_iou},
        }

    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def root():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


def app_factory() -> FastAPI:  # uvicorn --factory 用
    return create_app()
