"""分割 (configs/splits.json) の作成と前処理キャッシュ (data/processed/*.npz) の生成。

使い方: uv run python scripts/prepare.py
"""
import sys

sys.path.insert(0, "src")
from defect_hunter.data import load_config, make_splits, save_splits  # noqa: E402
from defect_hunter.dataset import build_cache  # noqa: E402


def main() -> None:
    cfg = load_config()
    splits = make_splits(cfg)
    save_splits(splits, cfg["splits_path"])
    hd = splits["has_defect"]
    for k in ("train", "val", "test"):
        n_pos = sum(hd.get(i, -1) == 1 for i in splits[k]) if k != "test" else "-"
        print(f"{k}: {len(splits[k])} 枚 (欠陥あり: {n_pos})")
    paths = build_cache(cfg, {k: splits[k] for k in ("train", "val", "test")})
    print({k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
