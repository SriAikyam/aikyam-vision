from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vision.config import CLIP_MODEL_ENABLED, CLIP_MODEL_NAME, CLIP_SCORE_THRESHOLD

# Text prompts per cluster for zero-shot CLIP classification.
# Each cluster has 3-5 descriptive prompts; the max similarity across prompts
# is taken as the cluster score so any single strong match wins.
CLUSTER_PROMPTS: dict[str, list[str]] = {
    "cluster_01": [
        "Shiva temple with lingam worship",
        "Nataraja dancing Shiva sculpture",
        "Mahadev trishul crescent moon",
        "Kedarnath mountain temple pilgrimage",
        "Shiva lingam with flowers and bilva leaves",
    ],
    "cluster_03": [
        "Krishna playing flute in Vrindavan forest",
        "Radha Krishna devotional painting",
        "Jagannath temple Puri chariot festival",
        "Bal Gopal butter stealing Krishna child",
        "Hare Krishna devotees dancing kirtan",
    ],
    "cluster_06": [
        "Tirupati Venkateswara temple gopuram",
        "Vishnu Lakshmi Narayana on Shesha",
        "Garuda Vishnu Sudarshana chakra",
        "Padmanabha reclining Vishnu",
    ],
    "cluster_15": [
        "Hanuman carrying Dronagiri mountain",
        "Bajrangbali Hanuman temple red flag",
        "Hanuman chalisa devotee",
        "Sankat Mochan temple Varanasi",
    ],
    "cluster_29": [
        "Ram temple Ayodhya golden spire",
        "Sita Ram Laxman Hanuman darbar",
        "Ramayana scene temple painting",
        "Ram navami celebration procession",
    ],
    "cluster_48": [
        "Ganesh Chaturthi idol immersion",
        "Siddhivinayak Ganesh temple",
        "elephant god Hindu puja",
        "Ganapati festival celebration",
    ],
    "cluster_49": [
        "Saraswati goddess veena white lotus",
        "Basant Panchami Saraswati puja yellow",
        "goddess of knowledge music learning",
    ],
    "cluster_50": [
        "Shirdi Sai Baba temple devotees",
        "Sai Baba dargah offering",
    ],
    "cluster_51": [
        "Sabarimala Ayyappa pilgrimage steps",
        "Makaravilakku festival Kerala",
        "Ayyappa devotees in black irumudi",
    ],
    "cluster_52": [
        "Murugan Karthikeya temple Tamil Nadu",
        "Vel Murugan Kavadi festival procession",
        "Palani Murugan hilltop temple",
    ],
    "cluster_53": [
        "Hindu puja aarti flame worship ceremony",
        "priest doing ritual offering coconut flowers",
        "temple aarti Ganga ghats evening",
    ],
    "cluster_54": [
        "Hindu temple gopuram stone tower",
        "ancient Indian temple carved architecture",
        "temple shikhara spire with sculptures",
    ],
    "cluster_55": [
        "Diwali festival diyas lights rangoli",
        "Holi colors powder festival celebration",
        "Navratri Durga festival garba dance",
        "Hindu religious festival crowd procession",
    ],
    "cluster_56": [
        "yoga meditation ashram sunrise",
        "Rishikesh yoga Ganges river bank",
        "pranayama breathing lotus pose meditation",
    ],
    "cluster_57": [
        "bhajan kirtan devotional singing group",
        "harmonium tabla Hindu devotional music",
        "satsang spiritual gathering singing hymns",
    ],
    "cluster_59": [
        "Char Dham pilgrimage Uttarakhand mountains",
        "Hindu pilgrims bathing holy river ghat",
        "Kumbh Mela Ganga sangam pilgrims",
    ],
    "cluster_60": [
        "Vedic learning Sanskrit scripture pandit",
        "guru teaching ashram disciples",
        "Hindu scripture reading recitation",
    ],
    "cluster_61": [
        "temple prasad laddoo blessed food",
        "modak sweet offering Ganesh",
        "langar community food religious",
    ],
    "cluster_63": [
        "Hindu astrology jyotish horoscope chart",
        "Panchang Hindu calendar almanac",
        "nakshatra zodiac Hindu astrology symbols",
    ],
}


@dataclass
class ClipResult:
    available: bool
    scores: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    error: str | None = None


class ClipModel:
    def __init__(self):
        self._processor = None
        self._model = None
        self._load_error: str | None = None

    def score_image(self, image_path: str | Path) -> ClipResult:
        if not CLIP_MODEL_ENABLED:
            return ClipResult(available=False, model_name="disabled")

        proc, model = self._load()
        if proc is None:
            return ClipResult(available=False, model_name=CLIP_MODEL_NAME, error=self._load_error)

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            scores: dict[str, float] = {}
            for cluster_id, prompts in CLUSTER_PROMPTS.items():
                inputs = proc(text=prompts, images=image, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=0).tolist()
                cluster_score = max(probs)
                if cluster_score >= CLIP_SCORE_THRESHOLD:
                    scores[cluster_id] = round(cluster_score, 4)

            return ClipResult(available=True, scores=scores, model_name=CLIP_MODEL_NAME)
        except Exception as exc:
            return ClipResult(available=False, model_name=CLIP_MODEL_NAME, error=str(exc))

    def _load(self):
        if self._processor is not None:
            return self._processor, self._model
        if self._load_error is not None:
            return None, None
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            self._model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            self._model.eval()
            return self._processor, self._model
        except Exception as exc:
            self._load_error = str(exc)
            return None, None

    def readiness(self) -> dict:
        if not CLIP_MODEL_ENABLED:
            return {"enabled": False, "available": False, "model": "disabled"}
        _, model = self._load()
        return {
            "enabled": True,
            "available": model is not None,
            "model": CLIP_MODEL_NAME,
            "error": self._load_error,
        }
