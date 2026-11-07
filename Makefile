.PHONY: test check prepare train summary assets serve

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

# ゲーム用素材 (画像・正解・AIの確率マップ) を outputs/game/ に生成
assets:
	uv run python scripts/build_game_assets.py

# ゲームを起動 -> http://127.0.0.1:8000
serve:
	uv run uvicorn defect_hunter.app:app_factory --factory --host 127.0.0.1 --port 8000

# eval は Phase 5 で追加する
