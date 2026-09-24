# ─────────────────────────────────────────────────────────────────────────────
# PDF Interleave + Rotate + 2-Up A4 Composer — Render-ready container
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── Runtime hygiene ─────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Bind address for Render ─────────────────────────────────────────────────
# Render sets PORT dynamically (default 10000). HOST must be 0.0.0.0 so the
# container accepts traffic from Render's edge proxy. The Python app already
# reads both from the environment, so no code change is needed.
ENV HOST=0.0.0.0 \
    PORT=10000

WORKDIR /app

# ── System deps (kept minimal for image size) ───────────────────────────────
# pypdf is pure Python, so no build toolchain is required.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# ── Python deps (separate layer for build-cache friendliness) ───────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application ─────────────────────────────────────────────────────────────
COPY main.py .

# ── Non-root user (Render runs containers as root by default; this is safer) ─
RUN useradd --create-home --shell /bin/bash appuser \
 && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

# ── Healthcheck so Render can detect a healthy boot ─────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# The app's __main__ block reads HOST/PORT from env and starts Uvicorn.
CMD ["python", "main.py"]
