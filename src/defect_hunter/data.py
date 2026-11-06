"""KSDD2 の読み込み・前処理・train/val 層化分割。

方針:
- 公式の train/test はそのまま使い、val は train からだけ作る (test への漏れを防ぐ)。
- 乱数は seed を引数で受け、同じ seed なら必ず同じ分割になるようにする。
"""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# "10301 (copy).png" のような混入ファイルを弾くため、5 桁数字だけを正規の画像とみなす。
_IMG_RE = re.compile(r"^(\d{5})\.png$")


def load_config(path: str | Path = "configs/data.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


@dataclass(frozen=True)
class Sample:
    id: str
    image_path: Path
    mask_path: Path


def list_samples(split_dir: str | Path) -> list[Sample]:
    """画像とマスクのペアを ID 順に列挙する。マスクが無い画像は黙って落とさずエラーにする。"""
    split_dir = Path(split_dir)
    samples = []
    for p in sorted(split_dir.iterdir()):
        m = _IMG_RE.match(p.name)
        if not m:
            continue
        mask = split_dir / f"{m.group(1)}_GT.png"
        if not mask.exists():
            raise FileNotFoundError(f"mask not found for {p}")
        samples.append(Sample(m.group(1), p, mask))
    return samples


def read_image(path: Path) -> np.ndarray:
    """BGR/グレースケールが混在しているため、必ず 3ch (RGB) uint8 に揃えて返す。"""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise OSError(f"cannot read image: {path}")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    elif im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGB)
    else:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def read_mask(path: Path) -> np.ndarray:
    """マスクを 0/1 の uint8 (H, W) で返す。"""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"cannot read mask: {path}")
    return (m > 0).astype(np.uint8)


def resize_pair(image: np.ndarray, mask: np.ndarray, height: int, width: int):
    """画像は INTER_AREA (縮小時にきれい)。マスクも面積補間してから >=0.5 で 2 値化する。

    最近傍補間だと、欠陥が数ピクセルしかない場合に縮小で消えやすいため。
    """
    img = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
    return img, (m >= 0.5).astype(np.uint8)


def stratified_split(labels: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """ラベル (0=良品, 1=欠陥) の比率を保ったまま train/val のインデックスに分ける。

    クラスごとに別々にシャッフルして val_fraction だけ val に回す。
    欠陥画像は少ない (約 10%) ので、単純な乱数分割だと val の欠陥数がぶれるため層化する。
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    labels_arr = np.asarray(labels)
    train_idx, val_idx = [], []
    for cls in (0, 1):
        idx = np.flatnonzero(labels_arr == cls)
        rng.shuffle(idx)
        # 少数クラスでも最低 1 枚は val に入れる
        n_val = max(1, round(len(idx) * val_fraction)) if len(idx) else 0
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    return sorted(train_idx), sorted(val_idx)


def has_defect(mask_path: Path) -> int:
    return int(read_mask(mask_path).any())


def make_splits(cfg: dict) -> dict:
    """train を train/val に層化分割し、test と合わせて ID リストの dict にする。"""
    raw = Path(cfg["raw_dir"])
    train_samples = list_samples(raw / "train")
    test_samples = list_samples(raw / "test")
    labels = [has_defect(s.mask_path) for s in train_samples]
    tr, va = stratified_split(labels, cfg["val_fraction"], cfg["seed"])
    return {
        "seed": cfg["seed"],
        "val_fraction": cfg["val_fraction"],
        "train": [train_samples[i].id for i in tr],
        "val": [train_samples[i].id for i in va],
        "test": [s.id for s in test_samples],
        "has_defect": {s.id: l for s, l in zip(train_samples, labels)},
    }


def save_splits(splits: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(splits, indent=1), encoding="utf-8")


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
