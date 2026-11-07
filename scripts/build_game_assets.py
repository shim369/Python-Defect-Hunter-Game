"""ゲーム用の素材を outputs/game/ に書き出す (画像・正解マスク・AI の確率マップ・index.json)。

データセットはライセンス上 Git に含めないので、各自のローカルで生成する。
使い方: uv run python scripts/build_game_assets.py [モデル名]   (既定: bce_dice_oversample)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from defect_hunter.data import load_config  # noqa: E402

# ゲームの AI 初期しきい値。val で選んだ値 (0.01) は F1 がほぼ平坦で偶然に近かったため、
# 慣例的で説明しやすい 0.5 に固定した。スライダーで 0.01〜0.99 に変更できる。
GAME_DEFAULT_THRESHOLD = 0.5


def main(model: str = "bce_dice_oversample", out: str = "outputs/game") -> None:
    cfg = load_config()
    res = json.loads(Path(f"results/{model}.json").read_text(encoding="utf-8"))
    d = np.load(Path(cfg["processed_dir"]) / "test.npz")
    probs = np.load(Path("outputs") / model / "test_probs.npy").astype(np.float32)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i, (id_, img, mask, p) in enumerate(zip(d["ids"], d["images"], d["masks"], probs)):
        id_ = str(id_)
        cv2.imwrite(str(out_dir / f"{id_}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out_dir / f"{id_}_gt.png"), mask * 255)
        cv2.imwrite(str(out_dir / f"{id_}_prob.png"), np.round(p * 255).astype(np.uint8))
        items.append({"id": id_, "has_defect": int(mask.any())})
    index = {"model": model, "threshold": GAME_DEFAULT_THRESHOLD, "min_area": res["min_area"],
             "height": int(d["images"].shape[1]), "width": int(d["images"].shape[2]), "items": items}
    (out_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"{len(items)} 枚 → {out_dir} (欠陥あり {sum(x['has_defect'] for x in items)}, thr={GAME_DEFAULT_THRESHOLD})")


if __name__ == "__main__":
    main(*sys.argv[1:2])
