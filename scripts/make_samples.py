"""README 用のサンプル図を作る (docs/samples/examples.png)。

選び方に恣意性が出ないよう、test の欠陥画像のうち ID が若い順に 3 枚を使う。
各列: 元画像 / 正解 (緑) / AI 予測 (青, しきい値 0.5)。
データはライセンス上、数枚のみ公開する (出典は README に明記)。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

G = Path("outputs/game")
idx = json.loads((G / "index.json").read_text())
ids = [i["id"] for i in idx["items"] if i["has_defect"]][:3]
thr = idx["threshold"]


def tint(img, mask, bgr):
    out = img.copy()
    out[mask] = (img[mask] * 0.4 + np.array(bgr) * 0.6).astype(np.uint8)
    return out


rows = []
for i in ids:
    im = cv2.imread(str(G / f"{i}.jpg"))
    gt = cv2.imread(str(G / f"{i}_gt.png"), 0) > 0
    pr = cv2.imread(str(G / f"{i}_prob.png"), 0).astype(np.float32) / 255 >= thr
    rows.append(np.hstack([im, tint(im, gt, (132, 220, 61)), tint(im, pr, (255, 179, 76))]))
sheet = np.hstack([np.pad(r, ((0, 0), (0, 8), (0, 0)), constant_values=20) for r in rows])
sheet = cv2.resize(sheet, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
Path("docs/samples").mkdir(parents=True, exist_ok=True)
cv2.imwrite("docs/samples/examples.png", sheet)
print(ids, sheet.shape)
