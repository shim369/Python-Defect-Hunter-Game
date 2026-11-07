"""2 つの結果 JSON が (許容誤差内で) 一致するか比べる。再現性チェック用。

使い方: uv run python scripts/compare_results.py A.json B.json [--tol 0]
tol=0 なら完全一致。別マシン/別 OS 間では浮動小数点の丸め差が出うるので、緩める場合は明示する。
"""
import argparse
import json
import sys


def flat(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{prefix}{k}.")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flat(v, f"{prefix}{i}.")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix.rstrip("."), float(x)


ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tol", type=float, default=0.0)
args = ap.parse_args()
A = dict(flat({k: v for k, v in json.load(open(args.a)).items() if k != "exp"}))
B = dict(flat({k: v for k, v in json.load(open(args.b)).items() if k != "exp"}))
if A.keys() != B.keys():
    sys.exit("FAIL: 結果の構造が違います")
worst = max((abs(A[k] - B[k]), k) for k in A)
print(f"{len(A)} 個の数値を比較: 最大差 {worst[0]:.3e} ({worst[1]})  許容 {args.tol:g}")
sys.exit(0 if worst[0] <= args.tol else 1)
