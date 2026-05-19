FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ML models (CLIP + Whisper) are opt-in. Set INSTALL_ML=true in production deployments.
ARG INSTALL_ML=false
COPY requirements-ml.txt .
RUN if [ "$INSTALL_ML" = "true" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi

COPY . .

# MODE=api starts FastAPI (HTTP analysis endpoints)
# MODE=worker starts Kafka consumer (post_created → simclusters)
ENV MODE=api
EXPOSE 8000

CMD if [ "$MODE" = "worker" ]; then \
      python main.py worker; \
    else \
      python main.py api; \
    fi
