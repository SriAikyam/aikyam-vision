FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libimage-exiftool-perl \
    ca-certificates \
    libsnappy-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Whisper (CPU-only torch ~300MB) — enable with INSTALL_WHISPER=true
ARG INSTALL_WHISPER=false
COPY requirements-whisper.txt .
RUN if [ "$INSTALL_WHISPER" = "true" ]; then pip install --no-cache-dir -r requirements-whisper.txt; fi

# CLIP (full torch + transformers ~2GB) — enable with INSTALL_CLIP=true
ARG INSTALL_CLIP=false
COPY requirements-clip.txt .
RUN if [ "$INSTALL_CLIP" = "true" ]; then pip install --no-cache-dir -r requirements-clip.txt; fi

COPY . .

# MODE=api   → FastAPI HTTP server (direct analysis endpoints)
# MODE=worker → Kafka consumer (post_created → simclusters)
ENV MODE=api
EXPOSE 8000

CMD if [ "$MODE" = "worker" ]; then \
      python main.py worker; \
    else \
      python main.py api; \
    fi
