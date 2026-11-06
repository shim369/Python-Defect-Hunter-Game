"""前処理済みキャッシュ (npz) の作成と、PyTorch Dataset。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import Sample, list_samples, read_image, read_mask, resize_pair


def build_cache(cfg: dict, ids_by_split: dict[str, list[str]]) -> dict[str, Path]:
    """画像を読み→リサイズ→uint8 のまま 1 ファイルに保存する。

    毎エポック PNG を読み直すと CPU 学習では遅いので、一度だけ処理して配列で持つ。
    uint8 のまま保存するのはサイズ節約のため (float に直すのは Dataset 側)。
    """
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    raw, out_dir = Path(cfg["raw_dir"]), Path(cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # val は train ディレクトリ、それ以外は同名ディレクトリから読む
    src_dir = {"train": "train", "val": "train", "test": "test"}
    paths = {}
    for split, ids in ids_by_split.items():
        by_id: dict[str, Sample] = {s.id: s for s in list_samples(raw / src_dir[split])}
        imgs, masks = [], []
        for i in ids:
            im, m = resize_pair(read_image(by_id[i].image_path), read_mask(by_id[i].mask_path), h, w)
            imgs.append(im)
            masks.append(m)
        path = out_dir / f"{split}.npz"
        np.savez_compressed(path, ids=np.array(ids), images=np.stack(imgs), masks=np.stack(masks))
        paths[split] = path
    return paths


class KSDD2Dataset(Dataset):
    """(image[3,H,W] float 0-1, mask[1,H,W] float 0/1) を返す。"""

    def __init__(self, npz_path: str | Path):
        d = np.load(npz_path)
        self.ids = d["ids"].tolist()
        self.images = d["images"]  # (N,H,W,3) uint8
        self.masks = d["masks"]  # (N,H,W) uint8

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        # PyTorch は (C,H,W) 順なので軸を入れ替える。0-1 に正規化すると学習が安定しやすい。
        x = torch.from_numpy(self.images[i]).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(self.masks[i]).unsqueeze(0).float()
        return x, y

    def labels(self) -> np.ndarray:
        """画像ごとの欠陥有無 (0/1)。サンプラーや層化の確認に使う。"""
        return (self.masks.reshape(len(self.masks), -1).max(axis=1) > 0).astype(int)
