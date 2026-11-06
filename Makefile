.PHONY: test check prepare

test:
	uv run pytest -q

# データ品質チェック (docs/data_quality.md を出力)
check:
	uv run python scripts/check_quality.py

# 分割 (configs/splits.json) と前処理キャッシュの作成
prepare:
	uv run python scripts/prepare.py

# train / eval / serve は Phase 3-5 で追加する
