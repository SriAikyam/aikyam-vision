import os


SIMCLUSTERS_BASE_URL = os.getenv("SIMCLUSTERS_BASE_URL", "http://localhost:8080")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "aikyam-vision-v1")
POST_CREATED_TOPIC = os.getenv("POST_CREATED_TOPIC", "post.created")

CLIP_MODEL_ENABLED = os.getenv("AIKYAM_CLIP_MODEL_ENABLED", "false").lower() == "true"
CLIP_MODEL_NAME = os.getenv("AIKYAM_CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

WHISPER_MODEL_ENABLED = os.getenv("AIKYAM_WHISPER_MODEL_ENABLED", "false").lower() == "true"
WHISPER_MODEL_SIZE = os.getenv("AIKYAM_WHISPER_MODEL_SIZE", "tiny")

# Min CLIP cosine similarity to count a cluster as matched
CLIP_SCORE_THRESHOLD = float(os.getenv("AIKYAM_CLIP_SCORE_THRESHOLD", "0.20"))

# Max frames extracted from video for CLIP scoring
VIDEO_FRAME_COUNT = int(os.getenv("AIKYAM_VIDEO_FRAME_COUNT", "5"))

HTTP_TIMEOUT_SECONDS = int(os.getenv("AIKYAM_HTTP_TIMEOUT_SECONDS", "10"))
