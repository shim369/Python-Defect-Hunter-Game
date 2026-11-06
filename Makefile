.PHONY: test lint

test:
	uv run pytest -q

# train / eval / serve は Phase 3-5 で追加する
