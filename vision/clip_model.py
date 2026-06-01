from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vision.config import CLIP_MODEL_ENABLED, CLIP_MODEL_NAME, CLIP_SCORE_THRESHOLD, CLIP_TOP_K

# Text prompts per cluster for zero-shot CLIP classification.
# Prompts are intentionally visual and discriminative. Broad devotional words
# made unrelated Hindu clusters score high during local testing.
CLUSTER_PROMPTS: dict[str, list[str]] = {
    "cluster_01": [
        "Figure with a crescent moon in matted hair holding a trident.",
        "Deity sitting on tiger skin with a snake around the neck.",
        "Cylindrical black stone lingam covered with green leaves and flowers.",
        "Bronze statue of a deity performing a cosmic dance inside a fire ring.",
        "Ascetic figure with three white ash stripes across the forehead."
    ],
    "cluster_03": [
        "Blue-skinned figure playing a bamboo flute with a peacock feather in hair.",
        "Couple featuring a dark-skinned male playing a flute and a female companion.",
        "Three wooden idols with large round eyes and brightly painted faces.",
        "Child figure with blue skin eating butter from a clay pot.",
        "Dark-skinned deity in yellow robes surrounded by white cows."
    ],
    "cluster_06": [
        "Four-armed deity holding a conch shell, spinning discus, mace, and lotus.",
        "Blue-skinned figure reclining on a giant multi-headed coiled serpent.",
        "Standing black stone idol with a large white V-shaped forehead mark.",
        "Four-armed figure riding a large humanoid eagle with wings.",
        "Venkateswara idol wearing a tall cylindrical gold crown and thick flower garlands."
    ],
    "cluster_15": [
        "Muscular monkey-faced figure holding a large golden mace.",
        "Figure with a monkey tail flying while carrying a rocky mountain peak.",
        "Monkey-faced deity covered entirely in bright orange vermilion paste.",
        "Kneeling monkey figure with hands folded in prayer tearing open his chest.",
        "Muscular figure with a monkey head wearing a golden crown and red dhoti."
    ],
    "cluster_29": [
        "Standing blue-skinned figure holding a large strung wooden bow and quiver.",
        "Group of three standing figures with bows and a kneeling monkey figure.",
        "Crowned figure in a dhoti holding a long arrow pointing to the ground.",
        "Male figure with a bow standing next to a woman in a red sari.",
        "Two men holding bows alongside a woman and a kneeling monkey."
    ],
    "cluster_48": [
        "Elephant-headed Ganesha figure with four arms.",
        "Ganesha holding round yellow sweets and a pink lotus.",
        "Seated elephant deity with a small grey mouse.",
        "Ganapati wearing a gold crown and marigold garlands.",
        "Elephant-headed figure with a red Om on the palm."
    ],
    "cluster_49": [
        "Figure in a white sari playing a stringed veena instrument.",
        "Deity holding a stringed instrument and a palm leaf manuscript.",
        "Figure in white clothing sitting on a white lotus next to a swan.",
        "Four-armed female figure holding a string of prayer beads and a book.",
        "Woman playing a long stringed musical instrument with carved gourds."
    ],
    "cluster_50": [
        "Older bearded man wearing a white cloth tied around his head.",
        "Seated bearded figure in a simple robe with right leg crossed over left.",
        "White marble statue of a bearded man sitting on a silver throne.",
        "Older man with a white beard wearing a long single-piece robe.",
        "Painted portrait of a bearded man with a headcloth raising one hand."
    ],
    "cluster_51": [
        "Seated youth with a meditation band tied around his folded knees.",
        "Devotees wearing black dhotis carrying two-compartment cloth bundles on heads.",
        "Barefoot men in dark clothes climbing steep stone steps in a forest.",
        "Figure sitting in a squatting pose with a belt wrapping the knees.",
        "Men with painted foreheads carrying dual-pouch cloth bundles on heads."
    ],
    "cluster_52": [
        "Standing male figure holding a long silver spear next to a peacock.",
        "Young boy deity holding a staff with a shaved head in a loincloth.",
        "Devotees carrying curved wooden arches heavily decorated with peacock feathers.",
        "Figure holding a spear standing on a pedestal between two female consorts.",
        "Procession balancing feathered semicircular wooden structures on their shoulders."
    ],
    "cluster_53": [
        "Hands holding a brass lamp with multiple lit camphor flames.",
        "Person pouring white milk from a brass pot over a stone idol.",
        "Priest waving a large tiered fire lamp in front of an altar.",
        "Hands placing red hibiscus flowers and unbroken coconuts on a brass plate.",
        "Group waving lit oil lamps in circles over a river at night."
    ],
    "cluster_54": [
        "Tall pyramidal temple tower covered in brightly painted tiered sculptures.",
        "Curved stone temple spire carved with intricate geometric patterns.",
        "Row of carved stone pillars lining a long shaded temple corridor.",
        "Exterior of a stone building with a stepped pyramidal roof.",
        "Large stone courtyard facing a tall gateway tower with statues."
    ],
    "cluster_55": [
        "People covered in bright pink and yellow colored powder throwing dye.",
        "Rows of small lit clay oil lamps arranged on the ground.",
        "Women in bright skirts dancing in a circle hitting wooden sticks together.",
        "Massive crowd pulling thick ropes attached to a large tall wooden chariot.",
        "Brightly colored geometric sand patterns drawn on the floor near lit lamps."
    ],
    "cluster_56": [
        "Person sitting cross-legged in a lotus pose with hands resting on knees.",
        "Group of people sitting on mats with closed eyes and straight backs.",
        "Figure doing breathing exercises with a finger blocking one nostril.",
        "Person performing physical stretching postures on a mat near a river.",
        "People in simple white clothes sitting in a circle inside a wooden hall."
    ],
    "cluster_57": [
        "Group of people sitting on the floor singing and clapping hands.",
        "Person pressing the keys of a wooden harmonium while singing.",
        "Hands playing a pair of small brass hand cymbals attached with string.",
        "People in a circle playing hand drums and singing with raised arms.",
        "Musician hitting a pair of tabla drums sitting on the floor."
    ],
    "cluster_59": [
        "Crowds of people bathing in a wide river next to stone steps.",
        "People walking on a mountain trail carrying decorated bamboo poles with water pots.",
        "Massive gathering of tents and people along a sandy riverbank.",
        "Pilgrims walking on a snowy mountain path toward a stone building.",
        "People standing waist-deep in river water with folded hands facing the sun."
    ],
    "cluster_60": [
        "Stack of ancient rectangular palm leaf manuscripts tied with string.",
        "Open book showing text written in the Devanagari script.",
        "Older man instructing young boys sitting cross-legged around a square fire pit.",
        "Priest wearing a white dhoti reading from a thick rectangular book.",
        "Hand pointing to horizontal lines of Sanskrit text on aged paper."
    ],
    "cluster_61": [
        "Brass plate filled with round yellow sweets and pieces of coconut.",
        "Steamed dumpling sweets shaped like teardrops arranged on a green leaf.",
        "Large steel buckets containing cooked rice and lentils for community feeding.",
        "Banana leaf serving plate topped with rice, fruits, and yellow sweets.",
        "Woven basket holding bananas, apples, and brightly colored square sweets."
    ],
    "cluster_63": [
        "Square grid chart with diagonal lines intersecting in the center.",
        "Printed almanac page displaying tables of numbers and Devanagari text.",
        "Hand drawing a diamond-shaped astrological birth chart on paper.",
        "Document showing a grid of twelve houses with planetary symbols.",
        "Circular diagram depicting zodiac signs and lunar phases on paper."
    ]
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

            # Score all prompts in one shared CLIP softmax. The previous approach
            # softmaxed prompts inside each cluster separately, so every cluster
            # could produce a high "best" score even for unrelated images.
            prompt_cluster_ids: list[str] = []
            prompts: list[str] = []
            for cluster_id, cluster_prompts in CLUSTER_PROMPTS.items():
                for prompt in cluster_prompts:
                    prompt_cluster_ids.append(cluster_id)
                    prompts.append(prompt)

            inputs = proc(text=prompts, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            prompt_probs = outputs.logits_per_image[0].softmax(dim=0).tolist()
            scores = _cluster_scores_from_prompt_probs(prompt_cluster_ids, prompt_probs)

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


def _cluster_scores_from_prompt_probs(
    prompt_cluster_ids: list[str],
    prompt_probs: list[float],
    threshold: float = CLIP_SCORE_THRESHOLD,
    top_k: int = CLIP_TOP_K,
) -> dict[str, float]:
    """Aggregate prompt probabilities into cluster scores and keep only top matches."""
    scores: dict[str, float] = {}
    for cluster_id, prob in zip(prompt_cluster_ids, prompt_probs):
        scores[cluster_id] = max(scores.get(cluster_id, 0.0), float(prob))

    ranked = sorted(
        (
            (cluster_id, round(score, 4))
            for cluster_id, score in scores.items()
            if score >= threshold
        ),
        key=lambda item: -item[1],
    )
    if top_k > 0:
        ranked = ranked[:top_k]
    return dict(ranked)
