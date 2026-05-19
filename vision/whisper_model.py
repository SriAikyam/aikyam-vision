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
                model_name=f"whisper-{WHISPER_MODEL_SIZE}",
                error=self._load_error,
            )

        try:
            result = model.transcribe(str(audio_path), task="transcribe")
            return WhisperResult(
                available=True,
                text=result.get("text", "").strip(),
                language=result.get("language", ""),
                model_name=f"whisper-{WHISPER_MODEL_SIZE}",
            )
        except Exception as exc:
            return WhisperResult(
                available=False,
                model_name=f"whisper-{WHISPER_MODEL_SIZE}",
                error=str(exc),
            )

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            return None
        try:
            import whisper
            self._model = whisper.load_model(WHISPER_MODEL_SIZE)
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
            "model": f"whisper-{WHISPER_MODEL_SIZE}",
            "error": self._load_error,
        }
