# Slim Debian Python base. The Microsoft Playwright image bundles Firefox +
# WebKit alongside Chromium (~600 MB) which this app never uses, so we install
# Chromium manually below instead.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_DIR=/opt/fastembed-cache \
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright-browsers \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    TASTE_AGENT_QUEUE_CONCURRENCY=8 \
    TASTE_AGENT_BROWSER_POOL_SIZE=3

WORKDIR /app

# Python deps first — own layer for cacheability. requirements.txt is auto-
# exported from uv.lock (`uv export --no-dev --no-hashes`) and pins every
# transitive dep with ==.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# We only ever launch headless (taste_agent/browser/backend.py:145 hard-codes
# headless=True), so we install only the chromium-headless-shell binary and
# its deps — skipping the full headed Chromium (~600 MB) and its extra deps.
# apt's default --install-recommends would also pile on fonts / xvfb / Mesa
# GL drivers / LLVM the headless shell never uses, so we disable that.
RUN echo 'APT::Install-Recommends "false";' > /etc/apt/apt.conf.d/99no-recommends \
    && apt-get update \
    && playwright install-deps chromium-headless-shell \
    && playwright install chromium-headless-shell \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pre-warm the fastembed ONNX model into the image so the first chat turn
# doesn't pay a download. Same MiniLM-L6-v2 model as before, ~80 MB on disk.
RUN python -c "from fastembed import TextEmbedding; \
TextEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2', cache_dir='/opt/fastembed-cache')"

# Source last — code edits don't bust the dep / browser / model layers.
COPY app.py ./
COPY taste_agent ./taste_agent/

EXPOSE 7860

# Python-based healthcheck so we don't need curl/wget in the final image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:7860/', timeout=4).status == 200 else 1)" \
    || exit 1

CMD ["python", "app.py"]
