# CPU 専用。Python は .python-version (3.15.1) を uv が取得するので、ホストと同じ版になる。
FROM debian:bookworm-slim

# make: Makefile のコマンドを同じように使うため / ca-certificates: Python 取得用
RUN apt-get update && apt-get install -y --no-install-recommends make ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python PYTHONUNBUFFERED=1

# 依存だけ先に入れる (ソースを変えても、この層のキャッシュが効く)
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile README.md ./
RUN uv sync --frozen

# データ (CC BY-NC-SA) はイメージに含めない。実行時に -v でマウントする (README 参照)。
EXPOSE 8000
# コンテナ外から繋げるよう 0.0.0.0 で待ち受ける
CMD ["uv", "run", "uvicorn", "defect_hunter.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
