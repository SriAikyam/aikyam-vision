from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vision.config import CLIP_MODEL_ENABLED, CLIP_MODEL_NAME, CLIP_SCORE_THRESHOLD, CLIP_TOP_K

# Text prompts per cluster for zero-shot CLIP classification.
# Prompts are intentionally visual and discriminative. Broad devotional words
# made unrelated Hindu clusters score high during local testing.
CLUSTER_PROMPTS: dict[str, list[str]] = {
    # ==========================================
    # PILLAR 1: DEVOTIONAL (SACRED HINDU CONTENT)
    # ==========================================
    "shiva_iconography": [
        "a photograph of a sacred Shivalinga idol decorated with bilva leaves and water offerings inside a temple",
        "a photograph of Lord Shiva meditating in a serene posture with a Trishul trident and crescent moon",
        "a sacred stone Shivalinga with milk abhishekam, flowers, and holy ash tripundra marking",
        "a small Shivalinga idol placed in a temple background",
        "a photo of Lord Shiva holding a trishul trident"
    ],

    "krishna_iconography": [
        "a photograph of the blue-skinned Hindu deity Lord Krishna playing a bamboo flute with a peacock feather",
        "a photograph of Radha and Krishna, featuring a dark-skinned male playing a flute with a female companion",
        "a photograph of a child Bal Krishna deity figure with blue skin eating white butter from a clay pot",
        "a photograph of the three traditional wooden Jagannath idols with large round eyes and painted faces",
        "a small idol of Lord Krishna placed in a temple background"
    ],

    "vishnu_iconography": [
        "a photograph of the four-armed Hindu deity Lord Vishnu holding a conch shell, sudarshana chakra, and mace",
        "a photograph of blue-skinned Lord Vishnu reclining on the giant multi-headed coiled serpent Sheshnag",
        "a photograph of a Lord Vishnu statue wearing a gold crown and fresh flower garlands",
        "a small idol of Lord Vishnu placed in a temple background",
        "a photo of Lord Vishnu inside a temple sanctum"
    ],

    "venkateswara_iconography": [
        "a photograph of a standing Tirupati Balaji idol with a large white V-shaped forehead tilak and tall gold crown",
        "a photograph of a Lord Venkateswara idol wearing a tall gold mukut crown and thick marigold flower garlands",
        "a photograph of a standing dark stone Tirupati Balaji idol inside a temple sanctum",
        "a small idol of Tirupati Balaji placed in a temple background",
        "a photo of Lord Venkateswara with white V tilak and garlands"
    ],

    "rama_iconography": [
        "a photograph of Lord Rama holding a large strung bow and arrow",
        "a photograph of a Ram Darbar idol set featuring Lord Rama, Goddess Sita, Lakshmana, and a kneeling Hanuman",
        "a photograph of the Ayodhya Ram Lalla idol adorned in gold holding a bow",
        "a small idol of Lord Rama placed in a temple background",
        "a photo of Shri Ram and Sita Devi adorned in royal traditional attire"
    ],

    "hanuman_iconography": [
        "a photograph of Lord Hanuman holding a golden gada mace in a devotional posture",
        "a photograph of a saffron-colored Lord Hanuman idol covered in bright orange sindoor carrying Sanjeevani mountain",
        "a temple idol of Bajrangbali Hanuman folded hands in devotion to Ram",
        "a small idol of Lord Hanuman placed in a temple background",
        "a photo of muscular monkey-faced Lord Hanuman wearing a gold crown"
    ],

    "ganesha_iconography": [
        "a photograph of a Lord Ganesha idol with elephant head, holding modak sweets and lotus flower",
        "a photograph of the elephant-headed Hindu deity Lord Ganesha with four arms sitting on a throne",
        "a photograph of a Ganapati idol wearing an ornate gold mukut crown and vibrant marigold flower garlands",
        "a small idol of Lord Ganesha placed in a temple background",
        "a photo of a seated elephant-headed deity Ganesha accompanied by mooshak mouse"
    ],

    "devi_iconography": [
        "a photograph of Goddess Durga riding a lion holding weapons in multiple arms",
        "a photograph of Goddess Lakshmi sitting on a pink lotus flower holding gold coins",
        "a photograph of Goddess Saraswati in a pure white sari playing a long stringed veena instrument",
        "a small idol of Goddess Durga, Lakshmi, or Saraswati in a temple background",
        "a photograph of a female Hindu goddess idol wearing an ornate gold crown and sari"
    ],

    "puja_rituals_and_temples": [
        "a close-up photograph of hands holding a brass ritual lamp with multiple lit burning camphor flames during aarti",
        "a photograph of a person pouring white milk from a brass pot over a stone deity idol during abhishekam ritual",
        "a photograph of an ancient stone Hindu temple gopuram tower with detailed traditional carvings",
        "a photograph of the sanctum sanctorum garbha griha of a Hindu temple illuminated by oil lamps",
        "a wide photograph of a large crowd of devotees standing in a river performing Chhath puja prayers with folded hands",
        "a photograph of devotees offering fruit and diya baskets on a riverside ghat during a religious ritual",
        "a photograph of covered metal processional utsava idols being carried in a temple festival procession",
        "a wide photograph of pilgrims queueing in a long line at a temple entrance gopuram for darshan",
        "a photograph of a Hindu temple exterior courtyard with devotees walking past ornately carved gateway towers",
        "a photograph of a priest performing a ritual with multiple devotees gathered around a temple altar"
    ],

    "festivals_and_celebrations": [
        "a photograph of lit brass diya oil lamps arranged on floor during Diwali festival",
        "a photograph of people celebrating Holi festival with bright organic gulal color powders",
        "a photograph of a colorful rangoli floor art design made with powders and flower petals"
    ],

    "safe_everyday_life": [
        "a normal everyday photograph of ordinary daily life with no religious or unsafe content",
        "a photograph of people walking on a city street, shops and vehicles in the background",
        "a portrait photograph of a person or small group in casual everyday clothing",
        "a photograph of a scenic mountain, river, or natural landscape with no people or buildings",
        "a photograph of a building or cityscape at night lit up with ordinary decorative or architectural lighting",
        "a photograph of a plate of vegetarian food such as pizza, pasta, salad, fries, or vegetable dishes"
    ],

    "modern_vehicles_and_tech": [
        "a photograph of a car, motorcycle, or other vehicle on a street or road",
        "a photograph of a modern smartphone, laptop, or other electronic device",
        "a photograph of a race car or sports car with visible sponsor logos and text",
        "a photograph of a bus, tram, or train with passengers boarding or riding",
        "a photograph of an old, rusted, or abandoned vehicle outdoors"
    ],

    "sports_and_recreation": [
        "a photograph of a floodlit sports stadium at night with a large cheering crowd and colorful advertising banners",
        "an action photograph of professional athletes competing on a field or court during a match",
        "a photograph of a cricket match in progress with players on a green pitch and a packed stadium",
        "a photograph of a football or soccer match with players and spectators in a stadium",
        "a photograph of people playing or watching an outdoor sports game in a casual daytime setting"
    ],

    "prasad_and_sacred_food": [
        "a photograph of temple mahaprasadam food served on fresh green banana leaf",
        "a photograph of sweet panchamrit and laddoo offered as holy prasadam in a temple"
    ],

    "non_vegetarian_food": [
        "a photograph of raw meat, butchery, butchered animal carcasses, raw chicken, raw beef, or pork cuts",
        "a photograph of cooked non-vegetarian meat dishes such as chicken curry, kebabs, or grilled meat",
        "a photograph of a hamburger or wrap with a visibly grilled beef, chicken, or lamb meat patty inside",
        "a photograph of seafood dishes, raw fish, shrimp, or a fish market display",
        "a photograph of fried or grilled chicken, wings, or drumsticks served as a meal"
    ],

    "christianity_iconography": [
        "a photograph of a church interior with pews, an altar, and stained glass windows",
        "a photograph of a statue or icon of Jesus Christ or the Virgin Mary",
        "a photograph of a wooden or metal crucifix or Christian cross displayed on a wall or altar",
        "a photograph of a priest or congregation during a Christian mass or communion service",
        "a photograph of a Christian nativity scene or Christmas manger display"
    ],

    "islam_iconography": [
        "a photograph of a mosque exterior with a dome and minaret towers",
        "a photograph of an open Quran with Arabic calligraphy text",
        "a photograph of worshippers praying on prayer mats inside a mosque",
        "a photograph of Islamic geometric or calligraphic art on a wall or building",
        "a photograph of a crescent moon and star symbol on a mosque or Islamic building"
    ],

    "buddhist_iconography": [
        "a photograph of a seated Buddha statue with legs crossed in meditation posture, elongated earlobes, and a serene expression",
        "a Buddha statue draped in a saffron or gold robe over one shoulder, seated cross-legged in a calm meditative pose",
        "a golden or bronze Buddha statue with a pointed cranial protrusion and half-closed eyes inside a temple or shrine",
        "a photograph of a Buddhist temple or pagoda with a large Buddha statue as the central figure",
        "a photograph of a standing or reclining Buddha statue in a Buddhist shrine"
    ],

    "sikh_iconography": [
        "a photograph of the Guru Granth Sahib holy scripture placed on a decorated platform under a canopy in a Sikh Gurdwara",
        "a photograph of a Sikh Gurdwara interior with a golden Khanda symbol of crossed swords and a circular chakkar displayed",
        "a photograph of a Sikh man wearing a turban and long beard inside a Gurdwara",
        "a photograph of the Ik Onkar symbol displayed on a wall or railing in a Sikh place of worship",
        "a photograph of a Gurdwara exterior with a white dome and an orange Nishan Sahib flag pole"
    ],

    # ==========================================
    # PILLAR 2: UNSAFE_CONTENT (SAFETY VIOLATIONS)
    # ==========================================
    "nudity_and_sexual_content": [
        "A fully nude person with no clothing covering the body.",
        "Explicit sexual activity or intimate physical contact between nude people.",
        "A close-up photograph focused on exposed genitals or explicit intimate body parts.",
        "Pornographic or sexually explicit photographic imagery.",
        "A person deliberately exposing intimate body parts in a sexually suggestive manner."
    ],

    "physical_violence_and_harm": [
        "two people engaged in a physical fight, aggressive postures, clenching fists, intense physical conflict",
        "a physical assault in progress, one person overpowering another, aggressive combat stance",
        "offensive body language, hostile framing with crude expressions and insulting hand motions",
        "public display of an obscene gesture, explicit vulgar hand sign captured clearly in public",
    ],

    "weapons_and_firearms": [
        "a close-up photograph of a handgun or pistol resting on a surface or held in a hand",
        "a photograph of a rifle or shotgun visible in the frame, held or resting against a wall",
        "a photograph of a large knife, machete, or sword blade held or displayed",
        "a photograph of a person holding or carrying a firearm",
        "a photograph of a collection of weapons such as guns and knives displayed together"
    ],

    "drugs_and_alcohol": [
        "a photograph of alcoholic beverage bottles or glasses of beer, wine, or liquor",
        "a photograph of a person drinking or holding an alcoholic drink at a bar or party",
        "a photograph of illegal drug paraphernalia such as pipes, syringes, or powder",
        "a photograph of a bar counter or nightclub with bottles of liquor and drinking glasses"
    ],

    "tobacco": [
        "a photograph of a cigarette pack or tobacco product box with a brand label, shown as a product photo",
        "a person holding a lit cigarette between their fingers close to their mouth, with visible smoke",
        "a close-up of a hand rolling or holding a hand-rolled cigarette, tobacco rolling paper visible",
        "a person smoking a cigar, pipe, or hookah with visible smoke",
        "a person lighting a cigarette with a lighter or match, cigarette held near the lips"
    ],

    "modern_nightlife_and_parties": [
        "a photograph of a crowd dancing inside a nightclub with flashing colored lights and a DJ booth",
        "a photograph of a party with loud music, a dance floor, and a large group of people dancing",
        "a photograph of a crowded rooftop or lounge bar at night with people drinking and socializing in a nightlife venue",
        "a photograph of a concert or rave with a stage, colored stage lighting, and a large dancing crowd packed together",
        "a photograph of a birthday or club party with balloons, a large group of people, and dancing"
    ],

    "commercial_spam_and_flyers": [
        "a photograph of a marketing flyer or banner with large discount codes and promotional text",
        "a photograph of an advertisement poster with a coupon code or limited time offer",
        "a screenshot of a promotional message advertising a product or service for sale",
        "a photograph of a shopping deal banner with percentage-off discount text",
        "a photograph of a QR code flyer offering a cash prize or lottery winning"
    ],

    "gambling": [
        "casino table, green felt surface layout, dealer area, card positions, and bets",
        "slot machine, glowing digital screen with spinning reels, betting buttons, casino floor",
        "person playing poker, holding playing cards close, thinking over chips on the table",
        "roulette wheel, spinning mechanical wheel with red and black numbered slots, ivory ball",
        "gambling chips on a table, stacks of colored betting tokens on casino felt layout",
        "sports betting screen, digital display board showing odds, point spreads, and live matches",
        "person placing a bet, sliding casino chips forward, handing money to a dealer slot",
        "casino interior, dimly lit room with flashing neon lights from rows of gambling machines",
        "blackjack game, cards dealt face up on felt, dealer hands visible, betting circles filled",
        "gambling activity, people rolling dice or wagering chips in a high-stakes environment",
    ],

    "medical": [
        "hospital emergency room, medical equipment monitors, gurneys, sterile lighting environment",
        "doctor treating a patient, healthcare provider wearing scrubs using medical instruments",
        "surgical procedure, surgeons wearing sterile gowns, masks, operating room overhead lights",
        "ambulance scene, emergency medical vehicle with flashing lights, open rear doors",
        "medical examination, patient sitting on a clinic table while a clinician checks vitals",
        "person receiving medical treatment, intravenous IV drip attached to an arm in a hospital",
        "operating room, clean medical theater with surgical lamps, instruments, and anesthesia carts",
        "healthcare workers assisting a patient, nurses and doctors collaborating around a bed",
        "hospital bed, electronic patient bed with clinical monitoring screens in the background",
        "clinical environment, sterile doctor's office, examination tools, and diagnostic charts",
    ],

    "child_safety": [
        "young child in a dangerous environment, toddler wandering unsupervised near hazards",
        "child near heavy traffic, small minor standing close to a busy road with passing cars",
        "child close to deep water without supervision, toddler standing on a dock or deep pool edge",
        "child in unsafe surroundings, young minor navigating structural ruins or hazardous debris",
        "lost child in a crowded place, crying toddler alone in a dense public area",
        "child requiring adult assistance, minor stuck in an unassisted precarious position",
        "child in an emergency situation, minor surrounded by first responders or crisis environments",
        "unsafe playground condition, child playing on broken or visibly compromised play structures",
        "child near hazardous equipment, minor close to heavy machinery or industrial tools",
        "minor in a risky environment, young individual exposed to heights, chemicals, or structural hazards",
    ],

    "animal_abuse": [
        "animal appearing injured, limping or wounded domestic pet or wildlife requiring medical aid",
        "animal in distress, trapped or confined creature showing clear physiological strain",
        "animal requiring rescue, stranded creature in a dangerous flood, ditch, or high ledge",
        "cruel treatment of an animal, visual context showing neglect, confinement, or physical harm",
        "animal in dangerous conditions, pet left in extreme elements or near hazardous debris",
        "injured wildlife, hurt deer, bird, or mammal showing clear physical trauma outdoors",
        "pet requiring emergency care, dog or cat being rushed into a veterinary clinic setting",
        "abandoned animal, stray dog or cat in poor condition wandering isolated urban streets",
        "animal rescue scene, welfare workers extracting a distressed animal from a hazardous area",
        "mistreatment of an animal, severe neglect or harmful handling captured visually",
    ],

    "nudity_high_level": [
        "person wearing swimwear, bikini or swim trunks, beach or poolside setting exposure",
        "person wearing underwear, brassiere, briefs, or boxers, revealing undergarment context",
        "person in revealing clothing, high skin exposure fashion, cropped or cut-out attire",
        "minimal clothing visible, adult subject with sparse or sheer fabric coverage",
        "shirtless adult, male or female subject with exposed torso, casual or athletic setting",
        "beach attire, sunbathers wearing minimal summer garments on sand with ocean background",
        "poolside attire, individuals in lounge wear or swimming suits next to water",
        "fashion photo with revealing clothing, artistic portrait prioritizing high skin visibility",
        "person wrapped in a towel, body covered by a white cotton sheet or towel post-shower",
        "adult with significant exposed skin, focus on bare back, torso, or limbs in a photo shoot",
    ],

    "ocr_offensive_text": [
        "offensive language visible on a sign, readable text displaying vulgarity or slurs",
        "hate message printed on a banner, clear text promoting discrimination against a group",
        "threatening text displayed, written warning indicating violence or physical intimidation",
        "abusive text in an image, legible digital font or print showing hostile insults",
        "offensive graffiti, spray-painted vulgar words or malicious statements on a concrete wall",
        "insulting words on clothing, printed graphic t-shirt displaying offensive profanity",
        "violent slogan on a poster, clear readable text calling for harmful actions or riots",
        "hostile text in a screenshot, legible text message dialogue containing severe abuse",
        "offensive words on a billboard, large public advertisement containing inappropriate terms",
        "profanity visible in printed text, close-up photograph of explicit derogatory language",
    ]
}



@dataclass
class ClipResult:
    available: bool
    scores: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    error: str | None = None
    image_embedding: list[float] | None = None


class ClipModel:
    def __init__(self):
        self._processor = None
        self._model = None
        self._load_error: str | None = None
        self._cached_prompt_cluster_ids: list[str] | None = None
        self._cached_text_features = None

    def _get_cached_text_features(self, proc, model):
        if self._cached_text_features is not None and self._cached_prompt_cluster_ids is not None:
            return self._cached_prompt_cluster_ids, self._cached_text_features

        import torch

        prompt_cluster_ids: list[str] = []
        prompts: list[str] = []
        for cluster_id, cluster_prompts in CLUSTER_PROMPTS.items():
            for prompt in cluster_prompts:
                prompt_cluster_ids.append(cluster_id)
                prompts.append(prompt)

        inputs = proc(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._cached_prompt_cluster_ids = prompt_cluster_ids
        self._cached_text_features = text_features
        return self._cached_prompt_cluster_ids, self._cached_text_features

    def score_image(self, image_path: str | Path) -> ClipResult:
        if not CLIP_MODEL_ENABLED:
            return ClipResult(available=False, model_name="disabled")

        proc, model = self._load()
        if proc is None or model is None:
            return ClipResult(available=False, model_name=CLIP_MODEL_NAME, error=self._load_error)

        try:
            import torch
            from PIL import Image

            image = Image.open(image_path).convert("RGB")

            # Retrieve pre-cached text features (computed once at startup)
            prompt_cluster_ids, text_features = self._get_cached_text_features(proc, model)

            inputs = proc(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Compute cosine similarity logits using fast PyTorch matrix multiplication
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * (image_features @ text_features.T)

            prompt_probs = logits_per_image[0].softmax(dim=0).tolist()
            scores = _cluster_scores_from_prompt_probs(prompt_cluster_ids, prompt_probs)
            image_embeds = image_features[0].tolist()
            return ClipResult(available=True, scores=scores, model_name=CLIP_MODEL_NAME, image_embedding=image_embeds)
        except Exception as exc:
            return ClipResult(available=False, model_name=CLIP_MODEL_NAME, error=str(exc))

    def _load(self):
        if self._processor is not None:
            return self._processor, self._model
        if self._load_error is not None:
            return None, None
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            # Optimize PyTorch CPU thread allocation (prevents OpenMP thread thrashing)
            torch.set_num_threads(4)

            self._processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=False)
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
    """Aggregate prompt probabilities into cluster scores and keep only top matches.

    Averages (not sums) each cluster's prompt probabilities -- summing would give
    clusters with more prompts more chances to accumulate softmax mass, biasing
    scores toward clusters with larger prompt lists regardless of image match quality.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for cluster_id, prob in zip(prompt_cluster_ids, prompt_probs):
        totals[cluster_id] = totals.get(cluster_id, 0.0) + float(prob)
        counts[cluster_id] = counts.get(cluster_id, 0) + 1
    scores = {cluster_id: total / counts[cluster_id] for cluster_id, total in totals.items()}

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
