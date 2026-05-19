from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision.config import WHISPER_MODEL_ENABLED, WHISPER_MODEL_SIZE


@dataclass
class WhisperResult:
    available: bool
    text: str = ""
    language: str = ""
    model_name: str = ""
    error: str | None = None


class WhisperModel:
    """
    Uses faster-whisper (CTranslate2 backend) — no PyTorch required,
    4x faster than openai-whisper on CPU, ~50MB install.
    """

    def __init__(self):
        self._model = None
        self._load_error: str | None = None

    def transcribe(self, audio_path: str | Path) -> WhisperResult:
        if not WHISPER_MODEL_ENABLED:
            return WhisperResult(available=False, model_name="disabled")

        model = self._load()
        if model is None:
            return WhisperResult(
                available=False,
                model_name=f"faster-whisper-{WHISPER_MODEL_SIZE}",
                error=self._load_error,
            )

        try:
            segments, info = model.transcribe(
                str(audio_path),
                task="transcribe",
                beam_size=1,           # fastest on CPU
                vad_filter=True,       # skip silence, speeds up religious content
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            return WhisperResult(
                available=True,
                text=text,
                language=info.language,
                model_name=f"faster-whisper-{WHISPER_MODEL_SIZE}",
            )
        except Exception as exc:
            return WhisperResult(
                available=False,
                model_name=f"faster-whisper-{WHISPER_MODEL_SIZE}",
                error=str(exc),
            )

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None
        try:
            from faster_whisper import WhisperModel as FasterWhisperModel
            self._model = FasterWhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8",   # quantised — fastest CPU inference
            )
            return self._model
        except Exception as exc:
            self._load_error = str(exc)
            return None

    def readiness(self) -> dict:
        if not WHISPER_MODEL_ENABLED:
            return {"enabled": False, "available": False, "model": "disabled"}
        model = self._load()
        return {
            "enabled": True,
            "available": model is not None,
            "model": f"faster-whisper-{WHISPER_MODEL_SIZE}",
            "error": self._load_error,
        }
