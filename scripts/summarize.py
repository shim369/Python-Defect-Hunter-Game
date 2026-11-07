"""results/*.json を比較表 (results/summary.md) にまとめる。

名前が *_s<数字> のものは同じ設定の別シードとみなし、設定ごとに平均±標準偏差を出す。
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

groups = defaultdict(list)
for p in sorted(Path("results").glob("*.json")):
    r = json.loads(p.read_text(encoding="utf-8"))
    groups[re.sub(r"_s\d+$", "", r["name"])].append(r)

KEYS = [("pixel", "iou"), ("pixel", "precision"), ("pixel", "recall"), ("pixel", "f1"),
        ("image", "precision"), ("image", "recall"), ("image", "f1")]


def ms(rs, k):
    v = np.array([r["test"][k[0]][k[1]] for r in rs])
    return f"{v.mean():.3f}" + (f"±{v.std(ddof=0):.3f}" if len(v) > 1 else "")


order = sorted(groups, key=lambda g: -np.mean([r["test"]["pixel"]["f1"] for r in groups[g]]))
L = ["| 設定 | シード数 | ピクセル IoU | ピクセル P | ピクセル R | ピクセル F1 | 画像 P | 画像 R | 画像 F1 |",
     "|---|---|---|---|---|---|---|---|---|"]
for g in order:
    rs = groups[g]
    L.append(f"| {g} | {len(rs)} | " + " | ".join(ms(rs, k) for k in KEYS) + " |")
L += ["", "値は test の平均±標準偏差 (シード数が 1 の設定は単一の値)。しきい値は各実行の val で決定。", ""]
Path("results/summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
