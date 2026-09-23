# IronSense backend. Build context is backend/ (the image never needs data/).
#
#   docker build -t ironsense-backend backend
#   docker run --rm -p 8000:8000 ironsense-backend
#
# For an Intel/AMD host (most EC2 instances) from an Apple Silicon Mac, add
# --platform linux/amd64. Render builds on its own Intel machines, so it needs nothing.
FROM python:3.12-slim

# libgomp1 is LightGBM's OpenMP runtime: without it `import lightgbm` fails at startup.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first, so a code change doesn't reinstall the ML stack.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app/paths.py resolves BACKEND_DIR to /app, so artifacts/ and seed_data/ sit beside app/.
COPY app ./app
COPY artifacts ./artifacts
COPY seed_data ./seed_data

# tools/ and training/ are deliberately absent: replay needs the 27 MB parquet from data/,
# and training belongs on a laptop, not in the serving image.

RUN useradd --create-home --uid 10001 ironsense \
    && mkdir -p /app/logs /data && chown -R ironsense /app /data
USER ironsense

# SQLite and raw logs go to /data so a mounted volume keeps them across restarts.
# Render's free tier has no disk: the DB is rebuilt from seed_data/ on every start.
ENV IRONSENSE_DB_PATH=/data/ironsense.db IRONSENSE_LOG_DIR=/data/logs PORT=8000

EXPOSE 8000

# Render (and most PaaS) inject $PORT; locally it falls back to 8000.
# One worker on purpose: machine state and the cab hub live in memory, so a second
# worker would hold a different copy and half the sockets would see stale state.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
