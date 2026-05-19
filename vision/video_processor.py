from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from vision.config import VIDEO_FRAME_COUNT


@dataclass
class VideoFrames:
    frame_paths: list[Path] = field(default_factory=list)
    audio_path: Path | None = None
    duration_seconds: float = 0.0
    error: str | None = None


def extract(video_path: str | Path, work_dir: str | Path | None = None) -> VideoFrames:
    """
    Extract VIDEO_FRAME_COUNT evenly-spaced frames and the audio track from a video.
    Uses system ffmpeg; returns empty result if ffmpeg is unavailable.
    """
    video_path = Path(video_path)
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="aikyam_vision_"))

    try:
        duration = _probe_duration(video_path)
        if duration <= 0:
            return VideoFrames(error="could not determine video duration")

        frame_paths: list[Path] = []
        for i in range(VIDEO_FRAME_COUNT):
            t = duration * (i / max(VIDEO_FRAME_COUNT - 1, 1))
            out = tmp / f"frame_{i:02d}.jpg"
            subprocess.run(
                ["ffmpeg", "-ss", str(t), "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "2", str(out), "-y"],
                capture_output=True, check=True, timeout=30,
            )
            if out.exists():
                frame_paths.append(out)

        audio_path: Path | None = None
        audio_out = tmp / "audio.wav"
        result = subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vn",
             "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
             str(audio_out), "-y"],
            capture_output=True, timeout=120,
        )
        if result.returncode == 0 and audio_out.exists():
            audio_path = audio_out

        return VideoFrames(
            frame_paths=frame_paths,
            audio_path=audio_path,
            duration_seconds=duration,
        )
    except FileNotFoundError:
        return VideoFrames(error="ffmpeg not found")
    except subprocess.TimeoutExpired:
        return VideoFrames(error="ffmpeg timed out")
    except Exception as exc:
        return VideoFrames(error=str(exc))


def _probe_duration(video_path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0
