.PHONY: test check prepare train summary

test:
	uv run pytest -q

# データ品質チェック (docs/data_quality.md を出力)
check:
	uv run python scripts/check_quality.py

# 分割 (configs/splits.json) と前処理キャッシュの作成
prepare:
	uv run python scripts/prepare.py

# configs/exp/*.toml を全て学習・評価する (4 並列、各 2 スレッド)。結果は results/*.json
train:
	ls configs/exp/*.toml | xargs -P 4 -I{} uv run python -m defect_hunter.train {}
	uv run python scripts/summarize.py

# 結果の比較表 (results/summary.md)
summary:
	uv run python scripts/summarize.py

# eval / serve は Phase 4-5 で追加する
