import os


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "aikyam-vision-v1")
POST_CREATED_TOPIC = os.getenv("POST_CREATED_TOPIC", "aikyam.post.created")
VISION_SCORES_TOPIC = os.getenv("VISION_SCORES_TOPIC", "aikyam.vision.scores")

CLIP_MODEL_ENABLED = os.getenv("AIKYAM_CLIP_MODEL_ENABLED", "false").lower() == "true"
CLIP_MODEL_NAME = os.getenv("AIKYAM_CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

WHISPER_MODEL_ENABLED = os.getenv("AIKYAM_WHISPER_MODEL_ENABLED", "false").lower() == "true"
WHISPER_MODEL_SIZE = os.getenv("AIKYAM_WHISPER_MODEL_SIZE", "tiny")

# Min shared-softmax CLIP score to count a cluster as matched.
# This is lower than the old per-cluster threshold because all prompts now compete together.
CLIP_SCORE_THRESHOLD = float(os.getenv("AIKYAM_CLIP_SCORE_THRESHOLD", "0.005"))
CLIP_TOP_K = int(os.getenv("AIKYAM_CLIP_TOP_K", "5"))

# Max frames extracted from video for CLIP scoring
VIDEO_FRAME_COUNT = int(os.getenv("AIKYAM_VIDEO_FRAME_COUNT", "5"))

HTTP_TIMEOUT_SECONDS = int(os.getenv("AIKYAM_HTTP_TIMEOUT_SECONDS", "10"))

MEDIA_SERVICE_URL = os.getenv("MEDIA_SERVICE_URL", "http://aikyam-media-control-plane:8080")
