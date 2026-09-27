# syntax=docker/dockerfile:1
#
# Multi-stage, non-root image. The base image is pinned by digest (resolved 2026-09-13); Dependabot's
# docker ecosystem proposes digest bumps. python:3.12-slim is multi-arch, so the index digest works
# on amd64 and arm64.
#
# uv is installed from PyPI at a pinned version inside the builder stage instead of being copied from
# ghcr.io/astral-sh/uv, so the build pulls from a single image registry. The extra registry was a
# second point of failure (networks that resolve Docker Hub but not ghcr.io could not build at all)
# and uv never reaches the runtime stage either way. Bump UV_VERSION by hand; Dependabot's uv
# ecosystem tracks uv.lock, not this build-time pin.

ARG PYTHON_IMAGE=python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea
ARG UV_VERSION=0.12.5

FROM ${PYTHON_IMAGE} AS builder
ARG UV_VERSION
RUN pip install --no-cache-dir --only-binary=:all: "uv==${UV_VERSION}"
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /build
# Dependencies first, so source edits do not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="deterministic-agent-simulator" \
      org.opencontainers.image.description="Read-only results API of the deterministic agent simulator; 'das campaign' runs simulations in the same image." \
      org.opencontainers.image.licenses="MIT"
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --home-dir /nonexistent \
       --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-m", "deterministic_agent_simulator.core.healthcheck"]
ENTRYPOINT ["python", "-m", "deterministic_agent_simulator"]
CMD ["serve"]
