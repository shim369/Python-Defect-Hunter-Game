.PHONY: test check prepare train eval summary assets serve repro-check docker-build docker-repro

QUICK = configs/quick/quick.toml
LIMIT = 128

test:
	uv run pytest -q

# データ品質チェック (docs/data_quality.md を出力)
check:
	uv run python scripts/check_quality.py

# 分割 (configs/splits.json) と前処理キャッシュ (data/processed/) の作成
prepare:
	uv run python scripts/prepare.py

# configs/exp/*.toml を全て学習・評価する (4 並列、各 2 スレッド)。結果は results/*.json
# 数時間かかる。短い動作確認は make repro-check
train:
	ls configs/exp/*.toml | xargs -P 4 -I{} uv run python -m defect_hunter.train {}
	uv run python scripts/summarize.py

# 保存済みチェックポイント (outputs/<name>/best.pt) から再評価し、比較表 results/summary.md を更新
eval:
	uv run python scripts/reeval.py
	uv run python scripts/summarize.py

summary:
	uv run python scripts/summarize.py

# ゲーム用素材 (画像・正解・AI の確率マップ) を outputs/game/ に生成
assets:
	uv run python scripts/build_game_assets.py

# ゲームを起動 -> http://127.0.0.1:8000
serve:
	uv run uvicorn defect_hunter.app:app_factory --factory --host 127.0.0.1 --port 8000

# 短い学習を同じ設定で 2 回走らせ、結果が完全に一致するか確かめる
repro-check:
	rm -rf outputs/repro_a outputs/repro_b
	uv run python -m defect_hunter.train $(QUICK) --limit $(LIMIT) --out outputs/repro_a --results outputs/repro_a/results
	uv run python -m defect_hunter.train $(QUICK) --limit $(LIMIT) --out outputs/repro_b --results outputs/repro_b/results
	uv run python scripts/compare_results.py outputs/repro_a/results/quick.json outputs/repro_b/results/quick.json

# --- Docker (CPU)。データはライセンス上イメージに含めず、ボリュームでマウントする ---
docker-build:
	docker build -t defect-hunter .

# コンテナ内で prepare → repro-check を実行する。結果 JSON はホストの outputs/docker/ に残す。
# data/raw だけを読み取り専用でマウントする (処理済みキャッシュはコンテナ内に作る)。
docker-repro:
	mkdir -p outputs/docker
	docker run --rm -v $(PWD)/data/raw:/app/data/raw:ro -v $(PWD)/outputs/docker:/app/outputs defect-hunter \
		sh -c "make prepare repro-check"
