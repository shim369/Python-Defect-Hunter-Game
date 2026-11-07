"""学習・評価の本体。設定ファイル (configs/exp/*.toml) 1 つ = 実験 1 つ。

使い方: uv run python -m defect_hunter.train configs/exp/bce_dice.toml
"""
from __future__ import annotations

import argparse
import json
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler

from .data import load_config
from .dataset import KSDD2Dataset
from .evaluate import best_threshold, evaluate, predict_probs
from .losses import build_loss
from .model import UNet


def seed_everything(seed: int) -> None:
    # 同じ設定なら同じ結果になるよう、全ての乱数源を固定する。
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(ds: KSDD2Dataset, cfg: dict, seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    if cfg.get("oversample", 0) > 0:
        # 欠陥画像を oversample 倍の確率で引く (不均衡対応: サンプリング側)
        w = np.where(ds.labels() == 1, float(cfg["oversample"]), 1.0)
        sampler = WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=g)
        return DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, generator=g)


def save_predictions(probs, ids, out_dir: Path, threshold: float) -> None:
    """確率マップ (0-255 の 8bit) と、しきい値で 2 値化した予測マスクを PNG で保存する。

    確率マップを残すのは、ゲームのしきい値スライダーをブラウザ側だけで動かすため。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for p, i in zip(probs, ids):
        Image.fromarray((p * 255).round().astype(np.uint8)).save(out_dir / f"{i}_prob.png")
        Image.fromarray(((p >= threshold) * 255).astype(np.uint8)).save(out_dir / f"{i}_pred.png")


def run(exp_path: str, data_cfg_path: str = "configs/data.toml", max_epochs: int | None = None,
        limit: int | None = None, out_root: str = "outputs", save_preds: bool = True,
        results_dir: str = "results") -> dict:
    exp = tomllib.loads(Path(exp_path).read_text(encoding="utf-8"))
    cfg = load_config(data_cfg_path)
    name = Path(exp_path).stem
    seed = exp.get("seed", cfg["seed"])
    seed_everything(seed)
    torch.set_num_threads(exp.get("threads", 4))

    proc = Path(cfg["processed_dir"])
    train_ds, val_ds, test_ds = (KSDD2Dataset(proc / f"{s}.npz") for s in ("train", "val", "test"))
    if limit:  # 動作確認用: 先頭だけ使う (欠陥画像が混ざるよう欠陥優先で選ぶ)
        keep = np.r_[np.flatnonzero(train_ds.labels())[: limit // 2], np.flatnonzero(train_ds.labels() == 0)[: limit // 2]]
        train_ds.images, train_ds.masks, train_ds.ids = (train_ds.images[keep], train_ds.masks[keep],
                                                         [train_ds.ids[i] for i in keep])
        val_ds.images, val_ds.masks = val_ds.images[:limit], val_ds.masks[:limit]
        test_ds.images, test_ds.masks, test_ds.ids = test_ds.images[:limit], test_ds.masks[:limit], test_ds.ids[:limit]

    model = UNet(base=exp.get("base", 16), depth=4)
    loss_fn = build_loss(exp["loss"], exp.get("pos_weight", 1.0))
    opt = torch.optim.AdamW(model.parameters(), lr=exp.get("lr", 1e-3), weight_decay=1e-4)
    loader = make_loader(train_ds, exp, seed)
    epochs = max_epochs or exp["epochs"]
    min_area = exp.get("min_area", 20)

    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
        # エポックごとに val で評価。しきい値は val 上の最良値で比較して、最良エポックを保存する
        thr, r = best_threshold(predict_probs(model, val_ds.images), val_ds.masks, min_area)
        history.append({"epoch": ep, "loss": tot / len(train_ds), "val_f1": r["pixel"]["f1"], "thr": thr})
        print(f"[{name}] ep{ep:02d} loss={tot/len(train_ds):.4f} val_pixF1={r['pixel']['f1']:.3f} "
              f"thr={thr:.2f} ({time.time()-t0:.0f}s)", flush=True)
        if r["pixel"]["f1"] > best_f1:
            best_f1, best_state = r["pixel"]["f1"], {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    val_probs = predict_probs(model, val_ds.images)
    thr, val_res = best_threshold(val_probs, val_ds.masks, min_area)  # しきい値は val だけで決める
    test_probs = predict_probs(model, test_ds.images)
    test_res = evaluate(test_probs, test_ds.masks, thr, min_area)
    result = {"name": name, "exp": exp, "threshold": thr, "min_area": min_area,
              "val": val_res, "test": test_res, "history": history}
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(results_dir, f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    np.save(out_dir / "test_probs.npy", test_probs.astype(np.float16))
    if save_preds:
        save_predictions(test_probs, test_ds.ids, out_dir / "predictions", thr)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--limit", type=int, help="動作確認用に使う枚数を絞る")
    ap.add_argument("--out", default="outputs", help="チェックポイントと予測の出力先")
    ap.add_argument("--results", default="results", help="結果 JSON の出力先")
    a = ap.parse_args()
    r = run(a.exp, max_epochs=a.epochs, limit=a.limit, out_root=a.out, results_dir=a.results)
    print(json.dumps({"threshold": r["threshold"], "test": r["test"]}, indent=1))
