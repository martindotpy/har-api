FROM python:3.12-alpine@sha256:9c51ecce261773a684c8345b2d4673700055c513b4d54bc0719337d3e4ee552e AS builder

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:3b898ca84fbe7628c5adcd836c1de78a0f1ded68344d019af8478d4358417399 /uv /bin/
RUN apk add g++ make

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Add cache for dependencies already compiled
ENV UV_CACHE_DIR=/tmp/uv-cache
RUN mkdir -p ${UV_CACHE_DIR}

RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --no-dev

FROM python:3.12-alpine@sha256:9c51ecce261773a684c8345b2d4673700055c513b4d54bc0719337d3e4ee552e AS runtime

WORKDIR /app

COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /app /app

COPY src src
COPY static static

ENV PATH="/app/.venv/bin:$PATH"
ENV HOST="0.0.0.0"

CMD ["uv", "run", "--no-dev", "src/main.py"]
