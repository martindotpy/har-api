FROM python:3.12-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db AS builder

COPY --from=ghcr.io/astral-sh/uv:latest@sha256:3b898ca84fbe7628c5adcd836c1de78a0f1ded68344d019af8478d4358417399 /uv /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Add cache for dependencies already compiled
ENV UV_CACHE_DIR=/tmp/uv-cache
RUN mkdir -p ${UV_CACHE_DIR}

RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --no-dev -v --no-build

FROM python:3.12-slim@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db AS runtime

WORKDIR /app

COPY --from=builder /bin/uv /bin/uv
COPY --from=builder /app /app

COPY src src
COPY static static
COPY build build

ENV PATH="/app/.venv/bin:$PATH"
ENV HOST="0.0.0.0"

CMD ["uv", "run", "--no-dev", "src/main.py"]
