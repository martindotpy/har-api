FROM python:3.13-alpine@sha256:18159b2be11db91f84b8f8f655cd860f805dbd9e49a583ddaac8ab39bf4fe1a7 AS builder

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:3b898ca84fbe7628c5adcd836c1de78a0f1ded68344d019af8478d4358417399 /uv /bin/
RUN apk add g++ make

WORKDIR /app

COPY pyproject.toml uv.lock ./

ENV UV_CACHE_DIR=/tmp/uv-cache
RUN mkdir -p ${UV_CACHE_DIR}

RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --no-dev


FROM python:3.13-alpine@sha256:18159b2be11db91f84b8f8f655cd860f805dbd9e49a583ddaac8ab39bf4fe1a7 AS runtime

WORKDIR /app

COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /app /app

COPY src src
RUN rm src/har_clustering.ipynb src/har_clustering.md.j2

ENV PATH="/app/.venv/bin:$PATH"

ENV HOST="0.0.0.0"

CMD ["uv", "run", "--no-dev", "src/main.py"]
