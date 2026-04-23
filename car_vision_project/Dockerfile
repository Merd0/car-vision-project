# syntax=docker/dockerfile:1.7

FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    VIRTUAL_ENV=/opt/venv

RUN python -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /build

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY car_vision_project/requirements.txt ./requirements.txt

RUN pip install --upgrade pip setuptools wheel \
    && pip install --requirement requirements.txt

COPY pyproject.toml .
COPY car_vision_project ./car_vision_project

RUN pip install --no-deps .


FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app \
    APP_ENV=production \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    UVICORN_WORKERS=1 \
    MODEL_CHECKPOINT_PATH=/app/artifacts/best_model.pt \
    MODEL_IMAGE_SIZE=224 \
    MODEL_DEVICE=cpu

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv
RUN mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail "http://127.0.0.1:${API_PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn car_vision_project.api.main:app --host ${API_HOST} --port ${API_PORT} --workers ${UVICORN_WORKERS}"]
