"""保存済みの best.pt から、しきい値探索をやり直して results/<name>.json を更新する (再学習なし)。

使い方: uv run python scripts/reeval.py [name ...]   (省略時は results/ の全て)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch  # noqa: E402

from defect_hunter.data import load_config  # noqa: E402
from defect_hunter.dataset import KSDD2Dataset  # noqa: E402
from defect_hunter.evaluate import best_threshold, evaluate, predict_probs  # noqa: E402
from defect_hunter.model import UNet  # noqa: E402
from defect_hunter.train import save_predictions  # noqa: E402


def main(names: list[str]) -> None:
    proc = Path(load_config()["processed_dir"])
    val, test = KSDD2Dataset(proc / "val.npz"), KSDD2Dataset(proc / "test.npz")
    for name in names:
        rp = Path(f"results/{name}.json")
        r = json.loads(rp.read_text(encoding="utf-8"))
        model = UNet(base=r["exp"].get("base", 16), depth=4)
        model.load_state_dict(torch.load(f"outputs/{name}/best.pt"))
        thr, val_res = best_threshold(predict_probs(model, val.images), val.masks, r["min_area"])
        probs = predict_probs(model, test.images)
        r.update(threshold=thr, val=val_res, test=evaluate(probs, test.masks, thr, r["min_area"]))
        rp.write_text(json.dumps(r, indent=1), encoding="utf-8")
        save_predictions(probs, test.ids, Path(f"outputs/{name}/predictions"), thr)
        print(f"{name}: thr={thr:.2f} val_pixF1={val_res['pixel']['f1']:.3f} test_pixF1={r['test']['pixel']['f1']:.3f}")


if __name__ == "__main__":
    main(sys.argv[1:] or [p.stem for p in sorted(Path("results").glob("*.json"))])
