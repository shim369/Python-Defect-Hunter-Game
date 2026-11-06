"""データ品質チェック: 破損、サイズ不一致、空マスク率、欠陥面積の分布、縮小後の欠陥消失。

使い方: uv run python scripts/check_quality.py  -> docs/data_quality.md を出力
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")
from defect_hunter.data import list_samples, load_config, read_image, read_mask, resize_pair  # noqa: E402


def check_split(name: str, split_dir: Path, h: int, w: int) -> tuple[list[str], dict]:
    problems, areas, shapes = [], [], []
    n = n_pos = gray = 0
    survive_any = survive_half = 0
    for s in list_samples(split_dir):
        n += 1
        try:
            raw = cv2.imread(str(s.image_path), cv2.IMREAD_UNCHANGED)
            if raw is not None and raw.ndim == 2:
                gray += 1
            img, mask = read_image(s.image_path), read_mask(s.mask_path)
        except OSError as e:  # 破損画像
            problems.append(f"{name}/{s.id}: 読み込み失敗 ({e})")
            continue
        if img.shape[:2] != mask.shape:
            problems.append(f"{name}/{s.id}: サイズ不一致 image{img.shape[:2]} vs mask{mask.shape}")
            continue
        shapes.append(img.shape[:2])
        if mask.any():
            n_pos += 1
            areas.append(int(mask.sum()))
            # 縮小しても欠陥が残るか (残らない = 学習・評価で見えない欠陥)
            _, m_small = resize_pair(img, mask, h, w)
            survive_half += int(m_small.any())
            m_f = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            survive_any += int((m_f > 0).any())
    a = np.array(areas) if areas else np.array([0])
    hs, ws = (np.array(x) for x in zip(*shapes)) if shapes else (np.array([0]), np.array([0]))
    stats = {
        "n": n, "pos": n_pos, "neg": n - n_pos, "empty_rate": (n - n_pos) / max(n, 1),
        "gray_images": gray,
        "h_range": (int(hs.min()), int(np.median(hs)), int(hs.max())),
        "w_range": (int(ws.min()), int(np.median(ws)), int(ws.max())),
        "area_pct": {p: int(np.percentile(a, p)) for p in (0, 10, 25, 50, 75, 90, 100)},
        "survive_half": survive_half, "survive_any": survive_any,
    }
    return problems, stats


def main() -> None:
    cfg = load_config()
    h, w = cfg["image"]["height"], cfg["image"]["width"]
    raw = Path(cfg["raw_dir"])
    lines = ["# データ品質チェック結果", "", f"縮小サイズ: {h}x{w} (HxW)", ""]
    all_problems = []
    for name in ("train", "test"):
        problems, s = check_split(name, raw / name, h, w)
        all_problems += problems
        lines += [
            f"## {name}", "",
            f"- 枚数: {s['n']} (欠陥あり {s['pos']} / 良品 {s['neg']})",
            f"- 空マスク率: {s['empty_rate']:.1%}",
            f"- グレースケール画像: {s['gray_images']} 枚 (読み込み時に 3ch 化)",
            f"- 高さ (min/median/max): {s['h_range']}, 幅: {s['w_range']}",
            f"- 欠陥面積 px (元解像度) パーセンタイル: {s['area_pct']}",
            f"- 縮小後も欠陥が残る数: {s['survive_half']}/{s['pos']} (>=0.5 で 2 値化)、"
            f"{s['survive_any']}/{s['pos']} (>0 で 2 値化した場合)", "",
        ]
    lines += ["## 問題点", ""] + ([f"- {p}" for p in all_problems] or ["- なし (破損・サイズ不一致ともに 0 件)"])
    out = Path("docs/data_quality.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
