import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    stream=sys.stdout,
)


def _api():
    import uvicorn
    uvicorn.run("api.routes:app", host="0.0.0.0", port=8000, log_config=None)


def _worker():
    from worker.consumer import VisionWorker
    VisionWorker().start()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"
    if mode == "worker":
        _worker()
    else:
        _api()
