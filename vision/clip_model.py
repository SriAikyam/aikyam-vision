from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from vision.config import CLIP_MODEL_ENABLED, CLIP_MODEL_NAME, CLIP_SCORE_THRESHOLD, CLIP_TOP_K

# Text prompts per cluster for zero-shot CLIP classification.
# Prompts are intentionally visual and discriminative. Broad devotional words
# made unrelated Hindu clusters score high during local testing.
CLUSTER_PROMPTS: dict[str, list[str]] = {
    # ==========================================
    # HINDU ICONOGRAPHY & TEMPLES (OPTIMIZED)
    # ==========================================
    "shiva_iconography": [
        "a photograph of the Hindu deity Lord Shiva with a crescent moon in matted hair holding a trident",
        "a photograph of Lord Shiva sitting cross-legged on a tiger skin with a cobra snake around his neck",
        "a photograph of a cylindrical black stone Shiva Lingam covered with bilva leaves and flowers",
        "a photograph of a bronze Nataraja statue performing the cosmic Tandava dance inside a ring of fire",
        "a photograph of a Hindu ascetic yogi with three horizontal white ash tilak stripes across the forehead"
    ],

    "krishna_iconography": [
        "a photograph of the blue-skinned Hindu deity Lord Krishna playing a bamboo flute with a peacock feather",
        "a photograph of Radha and Krishna, featuring a dark-skinned male playing a flute with a female companion",
        "a photograph of the three traditional wooden Jagannath idols with large round eyes and painted faces",
        "a photograph of a child Bal Krishna deity figure with blue skin eating white butter from a clay pot",
        "a photograph of Lord Krishna in bright yellow robes surrounded by grazing white cows in Vrindavan"
    ],

    "vishnu_iconography": [
        "a photograph of the four-armed Hindu deity Lord Vishnu holding a conch shell, sudarshana chakra, and mace",
        "a photograph of blue-skinned Lord Vishnu reclining on the giant multi-headed coiled serpent Sheshnag",
        "a photograph of the standing black stone Tirupati Balaji idol with a large white V-shaped forehead tilak",
        "a photograph of a four-armed deity figure riding the divine humanoid eagle Garuda with wide wings",
        "a photograph of a Lord Venkateswara idol wearing a tall gold crown and thick fresh flower garlands"
    ],

    "hanuman_iconography": [
        "a photograph of the muscular monkey-faced Hindu deity Lord Hanuman holding a large golden mace gada",
        "a photograph of Lord Hanuman flying through the air while carrying a rocky mountain peak in one hand",
        "a photograph of a monkey-faced Hanuman idol covered entirely in bright orange vermilion sindoor paste",
        "a photograph of a kneeling monkey deity Hanuman tearing open his chest to reveal Rama and Sita",
        "a photograph of a muscular figure with a monkey head wearing a golden crown and a traditional red dhoti"
    ],

    "rama_iconography": [
        "a photograph of the standing blue-skinned Hindu deity Lord Rama holding a large strung wooden bow",
        "a photograph of a Ram Darbar idol set with Rama, Lakshmana, Sita, and a kneeling Hanuman monkey figure",
        "a photograph of a crowned Lord Rama figure in a dhoti holding a long arrow pointing toward the ground",
        "a photograph of Lord Rama with a wooden bow standing next to Goddess Sita in a traditional red sari",
        "a photograph of two royal brothers holding bows alongside a standing woman and a kneeling monkey devotee"
    ],

    "ganesha_iconography": [
        "a photograph of the elephant-headed Hindu deity Lord Ganesha with four arms sitting on a throne",
        "a photograph of a Lord Ganesha idol holding round yellow laddu sweets and a pink lotus flower",
        "a photograph of a seated elephant deity Ganesha accompanied by his small mouse vehicle mooshak",
        "a photograph of a Ganapati idol wearing an ornate gold mukut crown and vibrant marigold flower garlands",
        "a photograph of an elephant-headed deity raising his palm displaying a red Om symbol on the skin"
    ],

    "saraswati_iconography": [
        "a photograph of the Hindu goddess Saraswati in a pure white sari playing a stringed veena instrument",
        "a photograph of Goddess Saraswati holding a traditional stringed instrument and a palm leaf manuscript text",
        "a photograph of a female deity in white clothing sitting on a large white lotus next to a white swan",
        "a photograph of a four-armed female goddess holding a string of japamala prayer beads and a sacred book",
        "a photograph of an altar featuring Goddess Saraswati playing a long stringed musical instrument with gourds"
    ],

    "sai_baba_iconography": [
        "a photograph of the Indian saint Shirdi Sai Baba wearing a white cloth tied around his head",
        "a photograph of a seated Shirdi Sai Baba figure in a simple robe with the right leg crossed over the left",
        "a photograph of a white marble statue of Shirdi Sai Baba sitting relaxed on an ornate silver throne",
        "a photograph of an older holy man with a white beard wearing a long single-piece kafni tunic robe",
        "a painted portrait photograph of Shirdi Sai Baba with a headcloth raising one hand in blessing"
    ],

    "ayyappan_sabarimala_pilgrimage": [
        "a photograph of the seated Hindu deity Lord Ayyappan with a meditation band tied around his folded knees",
        "a photograph of Sabarimala devotees wearing black dhotis carrying irumudi cloth bundles on their heads",
        "a photograph of barefoot male pilgrims in dark clothes climbing the steep stone steps of a forest path",
        "a photograph of an Ayyappan deity figure sitting in a squatting yoga posture with a knee belt",
        "a photograph of Ayyappa swami devotees with painted foreheads carrying dual-pouch cloth bundles on heads"
    ],

    "murugan_kartikeya_worship": [
        "a photograph of the standing young male Hindu deity Lord Murugan holding a long silver vel spear near a peacock",
        "a photograph of a young boy deity Palani Murugan holding a staff weapon with a completely shaved head",
        "a photograph of Hindu devotees carrying curved wooden kavadi arches decorated heavily with peacock feathers",
        "a photograph of Lord Kartikeya holding a spear standing on a stone pedestal between two female consorts",
        "a photograph of a Thaipusam religious procession balancing feathered semicircular wooden kavadi structures"
    ],

    "puja_rituals_and_offerings": [
        "a close-up photograph of hands holding a brass ritual lamp with multiple lit burning camphor flames during aarti",
        "a photograph of a person pouring white milk from a brass pot over a stone deity idol during abhishekam ritual",
        "a photograph of a Hindu priest waving a large multi-tiered fire lamp in front of a temple altar during puja",
        "a photograph of hands placing red hibiscus flowers and unbroken coconuts onto a decorative brass pooja thali",
        "a wide photograph of a group of people performing a Ganga Aarti waving lit oil lamps over a dark river at night"
    ],

    "hindu_temple_architecture": [
        "a photograph of a tall pyramidal South Indian gopuram temple tower covered in brightly painted tiered sculptures",
        "a photograph of a curved North Indian stone shikhara temple spire intricately carved with ancient patterns",
        "a photograph showing a row of beautifully carved stone pillars lining a long shaded temple corridor mandapa",
        "a photograph displaying the stone exterior of a traditional Hindu temple featuring a stepped pyramidal roof",
        "a wide photograph of a large open stone temple courtyard facing a tall gateway tower adorned with historical statues"
    ],

    "festivals_and_celebrations": [
        "a photograph of a joyful crowd covered in bright pink and yellow colored powder during the Holi festival",
        "a photograph showing long rows of small lit clay oil diyas arranged neatly on the ground during Diwali",
        "a photograph of women in bright traditional attire dancing in a circle hitting wooden sticks together during Navratri Dandiya",
        "a photograph of a massive crowd pulling thick ropes attached to a very large tall wooden chariot Ratha Yatra car",
        "a photograph of a brightly colored geometric rangoli sand pattern drawn on a floor surface near lit oil lamps"
    ],

    "yoga_and_meditation": [
        "a photograph of a person sitting cross-legged in a lotus pose performing dhyana meditation",
        "a photograph of a group of people sitting on yoga mats with closed eyes and straight aligned backs",
        "a photograph of an individual doing pranayama breathing exercises with a finger blocking one nostril",
        "a photograph of an individual performing physical stretching yoga postures on a mat outdoors near a river",
        "a photograph of people dressed in simple clothes sitting in a quiet circle inside a wooden meditation hall"
    ],

    "devotional_music_bhajan": [
        "a photograph of a group of people sitting gathered on the floor singing blankets and kirtans while clapping hands",
        "a photograph of a musician pressing the keys of a wooden harmonium box instrument while singing devotional music",
        "a close-up photograph of hands playing a pair of small brass manjira hand cymbals attached with string",
        "a photograph of people in a circle playing traditional dholak hand drums and singing with raised arms",
        "a photograph of a seated musician hitting a pair of traditional Indian tabla drums while sitting on the floor"
    ],

    "pilgrimage_and_holy_yatra": [
        "a wide photograph of crowds of Hindu pilgrims bathing in a wide sacred river next to historical stone ghat steps",
        "a photograph of Kanwar Yatra pilgrims walking on a mountain trail carrying decorated bamboo poles with water pots",
        "a wide aerial photograph of the massive Kumbh Mela gathering of tents and crowds along a sandy riverbank",
        "a photograph of pilgrims walking along a snowy mountain path directly toward an ancient stone Himalayan temple",
        "a photograph of devotees standing waist-deep in moving river water with folded hands performing Surya Arghya to the sun"
    ],

    "vedic_scriptures_and_education": [
        "a photograph of a stacked bundle of ancient rectangular palm leaf manuscripts written in old Sanskrit script",
        "a photograph of an open book displaying sacred Vedic texts written legibly in the traditional Devanagari script",
        "a photograph of a Hindu guru instructing young disciples sitting cross-legged around a square homa fire pit",
        "a photograph of a pandit priest wearing a white cotton dhoti reading attentively from a thick religious book",
        "a close-up photograph of a hand pointing explicitly to horizontal lines of Sanskrit text on aged yellow paper"
    ],

    "prasad_and_sacred_food": [
        "a photograph of a shiny brass plate filled completely with round yellow laddu sweets offered as prasad",
        "a photograph of cooked modak dumpling sweets shaped like teardrops arranged neatly on a fresh green leaf",
        "a photograph of large industrial steel containers containing bhandara food cooked for community feeding",
        "a photograph of a green banana leaf serving plate topped with white rice, traditional dishes, and sweet prasad",
        "a photograph of a woven basket holding fresh bananas, coconuts, and brightly colored square sweets for offering"
    ],

    "astrology_and_almanac_jyotish": [
        "a photograph of a Vedic astrology square grid chart layout with diagonal lines intersecting in the center",
        "a photograph of a printed Hindu panchang almanac page displaying complex tables of numbers and Devanagari text",
        "a close-up photograph of an astrologer hand drawing a diamond-shaped kundali birth chart onto paper sheets",
        "a photograph of a document diagram showing a grid of twelve houses containing traditional planetary symbols",
        "a photograph of a circular astronomical diagram depicting rashi zodiac signs and various lunar phases on paper"
    ],

    # ==========================================
    # SECULAR, SECURE & EVERYDAY CONTENT (NON-RELIGIOUS)
    # ==========================================
    "safe_everyday_life": [
        "a normal everyday photograph with clear daytime lighting and realistic colors",
        "a photograph of a standard family gathering in a cozy domestic living room setting",
        "a portrait photograph of random people smiling together with clear facial expressions",
        "a casual photograph of friends taking a selfie together in an outdoor urban setting",
        "a photograph of a group of corporate coworkers standing together in a modern office space",
        "a wide photograph of a school classroom filled with students sitting attentively at desks",
        "a daylight photograph of citizens walking along a paved path in a public green park",
        "a serene landscape photograph of an ordinary peaceful outdoor nature scene",
        "a photograph of a person cooking food in a domestic modern kitchen setup",
        "a professional wide landscape photograph of mountains or oceans under a clear blue sky",
        "a wide daytime photograph of pedestrians and public vehicles on a busy city street",
        "a photograph of a person casually typing and working on a laptop at a workspace desk",
        "an ordinary indoor photograph of a standard residential living room arrangement"
    ],

    "secular_education_and_tech": [
        "a photograph of an instructor teaching physics formulas on a green chalkboard to students",
        "a photograph of software engineers analyzing code lines on dual computer monitors",
        "a photograph of complex server racks with flashing green lights inside a data center",
        "a photograph of an open academic textbook showing graphs and english text on a table",
        "a photograph of a student studying quietly inside a modern library book aisle"
    ],

    "sports_and_recreation": [
        "an action photograph of professional athletes competing at a crowded soccer match stadium",
        "a photograph of runners pacing themselves on a running track during a marathon event",
        "a photograph of a person playing tennis hitting a yellow ball over the court net",
        "a photograph of swimmers diving into a clean indoor public swimming pool lane",
        "a photograph of people riding bicycles along a marked asphalt bike path in daylight"
    ],

    "medical_and_healthcare": [
        "a photograph of a sterile hospital emergency room equipped with electronic health monitors",
        "a photograph of a medical doctor in surgical scrubs actively treating an adult patient",
        "a photograph of an ongoing clean surgical procedure inside a professional operating room",
        "a photograph of a clinical medical examination happening inside a doctor's clinic office",
        "a photograph of an empty clinical hospital bed surrounded by static medical machines"
    ],

    # --- DEITY ICONOGRAPHY CATEGORIES ---
    "shiva_iconography": [
        "a photograph of a sacred Shivalinga idol decorated with bilva leaves and water offerings inside a temple",
        "a photograph of Lord Shiva meditating in a serene posture with a Trishul trident and crescent moon",
        "a sacred stone Shivalinga with milk abhishekam, flowers, and holy ash tripundra marking"
    ],

    "ganesha_iconography": [
        "a photograph of a Lord Ganesha idol with elephant head, holding modak sweets and lotus flower",
        "a decorated Ganesha idol during Ganesh Chaturthi festival with modak offerings and marigold garlands",
        "a carved stone or brass murti of Lord Ganesha in a temple shrine"
    ],

    "vishnu_krishna_iconography": [
        "a photograph of Lord Krishna playing a bamboo flute bansuri with a peacock feather mor pankh in his hair",
        "a photograph of Lord Vishnu holding shankha conch shell and sudarshan chakra disc in a temple shrine",
        "a beautifully adorned idol of Radha Krishna dressed in colorful silk attire and flower garlands"
    ],

    "devi_durga_iconography": [
        "a photograph of Goddess Durga riding a lion holding sacred weapons in her multiple hands",
        "a photograph of Goddess Lakshmi seated on a pink lotus flower with gold coins",
        "a photograph of Goddess Saraswati playing the veena instrument surrounded by sacred books"
    ],

    "hanuman_iconography": [
        "a photograph of Lord Hanuman holding a gada mace in a devotional posture",
        "a photograph of a saffron-colored Lord Hanuman idol carrying the Sanjeevani mountain",
        "a temple idol of Bajrangbali Hanuman folded hands in devotion to Ram"
    ],

    "ram_sita_iconography": [
        "a photograph of Lord Ram holding a bow dhanush standing beside Sita and Lakshman",
        "a photograph of the grand Ayodhya Ram Mandir temple or Ram Lalla idol adorned in gold",
        "a sacred deity idol of Shri Ram and Sita Devi inside a temple sanctum"
    ],

    # --- DEVOTIONAL & RITUAL CATEGORIES ---
    "temple_architecture_and_deities": [
        "a photograph of an ancient stone Hindu temple gopuram tower with detailed traditional carvings",
        "a photograph of the sanctum sanctorum garbha griha of a Hindu temple illuminated by oil lamps"
    ],

    "puja_and_aarti_rituals": [
        "a photograph of a brass aarti lamp with burning cotton wicks held during temple evening aarti",
        "a photograph of a traditional havan kund fire ritual with priest pouring ghee offerings into sacred fire"
    ],

    "bhajan_and_kirtan": [
        "a photograph of devotees singing bhajan kirtan playing dholak drum, tabla, and harmonium",
        "a group of devotees clapping hands and chanting devotional songs in a temple hall"
    ],

    "festival_celebrations": [
        "a photograph of lit brass diya oil lamps arranged on floor during Diwali festival",
        "a photograph of people celebrating Holi festival with bright organic gulal color powders"
    ],

    "sacred_yatra_and_pilgrimage": [
        "a photograph of the sacred Ganga aarti at Varanasi ghats along the holy river Ganges",
        "a photograph of Kedarnath temple surrounded by snow-capped Himalayan mountain peaks"
    ],

    "vedic_wisdom_and_astrology": [
        "a photograph of a traditional Hindu panchang astrology chart written on ancient paper",
        "a photograph of sacred Sanskrit slokas written in Devanagari script on a manuscript"
    ],

    "prasadam_and_sattvic_food": [
        "a photograph of temple mahaprasadam food served on fresh green banana leaf",
        "a photograph of sweet panchamrit and laddoo offered as holy prasadam in a temple"
    ],

    "spiritual_gurus_and_discourses": [
        "a photograph of a revered Hindu spiritual guru in saffron robes giving a Ramkatha discourse",
        "a photograph of a sadhu or sanyasi in orange robes seated in peaceful meditation"
    ],

    "vedic_education_and_gurukul": [
        "a photograph of young gurukul students wearing traditional dhoti reciting Vedic slokas",
        "a classroom in a traditional gurukul with teacher teaching Sanskrit scriptures"
    ],

    "classical_temple_arts": [
        "a photograph of a classical Bharatanatyam or Kathak dancer in traditional temple dance costume",
        "a photograph of a priest blowing a sacred shankha conch shell during temple rituals"
    ],

    "sacred_art_and_rangoli": [
        "a photograph of a colorful rangoli floor art design made with powders and flower petals",
        "a photograph of a sacred Swastika or Om symbol drawn with holy chandan and kumkum"
    ],

    "temple_announcements_and_seva": [
        "a photograph of volunteers distributing free anna daan meals to devotees at a temple seva hall",
        "a photograph of Gau Seva workers feeding sacred cows at a temple gaushala shelter"
    ],

    # --- PROHIBITED & SAFETY CATEGORIES ---
    "nudity_and_sexual_content": [
        "Nude or completely unclothed figures displaying explicit or intimate suggestive poses.",
        "Inappropriate exposure, revealing undergarments, or explicit nudity."
    ],

    "child_safety_risk": [
        "Unsupervised young child or toddler wandering near dangerous road traffic or deep water hazards.",
        "Minor child in an unassisted hazardous emergency situation."
    ],

    "non_veg_and_meat": [
        "Raw meat, butchery, butchered animal carcasses, raw chicken, raw beef, or pork cuts.",
        "Cooked non-vegetarian meat dishes, seafood, or raw fish market display."
    ],

    "scam_and_phishing": [
        "Scam flyer with personal UPI QR codes asking for money or fake cash prizes.",
        "Phishing banner promising guaranteed lottery winnings or fraudulent financial schemes."
    ],

    "fake_babas_and_occult_scams": [
        "Occult vashikaran ad promising black magic cures, love back guarantees, or superstitious spells.",
        "Fraudulent black magic practitioner advertisement banner on social media."
    ],

    "disrespectful_temple_acts": [
        "Inappropriate pop dancing inside a Hindu temple sanctum or climbing disrespectfully on sacred idols.",
        "Mocking religious rituals or desecrating sacred temple space."
    ],

    "footwear_in_sacred_space": [
        "Wearing outdoor leather shoes, boots, or sneakers inside a Hindu temple sanctum near sacred idols.",
        "Footwear placed directly on sacred temple steps or near holy altar."
    ],

    "substance_alcohol_drugs": [
        "Alcoholic beverage bottles, beer glasses, pub drinking, or illegal narcotics.",
        "Cigarette smoking, tobacco smoke, or substance abuse."
    ],

    "spam_and_promotions": [
        "Graphic flyer with large marketing text, coupon codes, discount banners, and commercial promotional offers.",
        "Commercial advertisement, shopping deals, buy now banner, promotional poster on screen."
    ],

    "violence_and_harm": [
        "Close-up of a handgun, pistol, knife, or weapon resting on a surface or held menacingly.",
        "Bright red blood on skin, bandages, violent physical assault, or severe medical wounds."
    ],

    "hate_speech_and_abuse": [
        "Offensive text, vulgar slurs, abusive profanity, or hate speech printed on signs, banners, or clothing.",
        "Discriminatory text message, hostile slurs, or abusive written dialogue."
    ],

    "animal_abuse": [
        "Cruel treatment of an animal, severe neglect, physical harm, or confined distressed pet."
    ],

    "ocr_offensive_text": [
        "Offensive language visible on a sign, readable text displaying vulgarity or slurs."
    ],

    # --- GENERAL RELEVANCE CATEGORIES ---
    "modern_vehicles": [
        "a photograph of a modern passenger car sedan driving along a clean asphalt road highway",
        "a photograph of a two-wheeled motorcycle or motor scooter parked next to a city street curb",
        "a photograph of a modern SUV sports utility vehicle or truck traveling on an open highway"
    ],

    "modern_nightlife_and_parties": [
        "a photograph of a massive crowd of young people dancing inside a dimly lit club with flashing colored neon lights",
        "a photograph of a group of friends holding up drink glasses and laughing together inside a modern restaurant or pub"
    ],

    "medical_and_healthcare": [
        "a photograph of a sterile hospital emergency room equipped with electronic health monitors"
    ],

    "everyday_safe_life": [
        "a family gathering, multiple generations smiling, cozy living room background, warm lighting",
        "people smiling together, high-quality portrait photograph, happy expressions, clear facial features"
    ],
"safe": [
        "a normal everyday photograph, clear daylight, realistic colors, candid composition",
        "a family gathering, multiple generations smiling, cozy living room background, warm lighting",
        "people smiling together, high-quality portrait photograph, happy expressions, clear facial features",
        "friends taking a selfie, holding a smartphone, laughing, casual outdoor environment",
        "a group of coworkers in an office, professional attire, standing together, modern workspace background",
        "a classroom with students, sitting at desks, school environment, teacher at the whiteboard",
        "people walking in a park, green trees, sunny day, casual walking path background",
        "a peaceful outdoor scene, calm nature landscape, scenic view, serene environment",
        "a person cooking in a kitchen, standing near a stove, chopping vegetables, domestic home setting",
        "a pet sitting indoors, domestic cat or dog looking at the camera, cozy room floor",
        "a landscape photograph, wide shot of mountains or oceans, clear sky, professional nature photography",
        "a city street during daytime, pedestrians walking on sidewalks, cars on the road, urban architecture",
        "a nature trail, hiking path surrounded by trees and foliage, daylight filtering through leaves",
        "a birthday celebration, lit candles on a cake, people gathered around, colorful party decorations",
        "a wedding ceremony, bride and groom at the altar, guests watching, elegant formal venue",
        "a sports event, athletes competing on a field or court, stadium seating, daytime action shot",
        "a child playing with toys, sitting on a colorful playroom rug, toys scattered around",
        "a person reading a book, seated comfortably on an armchair, relaxed posture, indoor lighting",
        "a person working on a laptop, sitting at a desk, typing, focused expression, office or home office setup",
        "an ordinary indoor scene, living room or bedroom view, standard furniture arrangement, casual home environment",
    ],

    "physical_violence": [
        "two people engaged in a physical fight, aggressive postures, clenching fists, intense physical conflict",
        "a person punching another person, striking with a fist, physical altercation, close impact shot",
        "a person kicking another individual, aggressive leg strike, forceful physical conflict",
        "a physical assault in progress, one person overpowering another, aggressive combat stance",
        "someone attacking another person, violent physical confrontation, dynamic action shot",
        "violent confrontation between people, chaotic physical struggle, aggressive gestures and positioning",
        "a person being physically assaulted, defensive posture, active physical attack occurring",
        "aggressive physical altercation, street scuffle, close contact physical violence",
        "street fight between individuals, outdoor public setting, raw physical conflict",
        "violent attack captured in a photograph, motion blur, aggressive movements, physical clashing",
        "close-up of a physical assault, tight framing on an aggressive physical struggle",
        "crowd involved in violent fighting, chaotic brawl, multiple individuals clashing physically",
        "violent conflict outdoors, physical altercation on a sidewalk or public park",
        "violent confrontation indoors, room setting with people engaged in a physical struggle",
        "person pushing another aggressively, shoving motion, hostile physical contact",
        "violent struggle between two people, grappling, wrestling, intense physical dispute",
        "group involved in physical violence, multiple people fighting, aggressive group confrontation",
        "assault occurring in public, outdoor space, onlookers watching a physical fight",
        "violent behavior toward another person, physical strike or slam, aggressive movement",
        "person striking another person, hand contact or physical blow during an altercation",
    ],

    "blood_injury": [
        "visible blood on a person's clothing, red stains on fabric, aftermath of an injury",
        "visible bleeding injury, open cut or wound with red blood flow, medical context",
        "person with an injured arm, wrapped in temporary bandage or showing trauma",
        "person with facial injuries, bruises, scrapes, or lacerations visible on skin",
        "blood visible on the ground, red drops or pools on concrete or flooring surface",
        "medical emergency involving injuries, first responders present, visible trauma",
        "injured individual requiring assistance, person holding an injured limb, distressed expression",
        "person with visible wounds, cuts, abrasions, or bodily injury from an accident",
        "accident victim with injuries, lying down or being treated, visible signs of physical trauma",
        "emergency scene with injured person, paramedics attending to a patient with injuries",
        "injured athlete during competition, lying on the field, medical staff attending to a wound",
        "visible injury after an accident, physical trauma visible on skin or limbs",
        "person receiving first aid, bandage being applied to a visible injury or cut",
        "injured pedestrian, sitting or lying on the street, receiving emergency attention",
        "visible bleeding after an accident, red fluid pooling near an injured individual",
    ],

    "graphic_injury": [
        "graphic injury scene, severe medical emergency, extensive physical trauma visible",
        "severe traumatic injury, catastrophic bodily damage, emergency medical context",
        "disturbing accident scene, severe impact aftermath, extensive trauma and damage",
        "graphic emergency scene, severe bodily harm, critical medical crisis",
        "extreme physical trauma, severe structural damage to tissue or limbs",
        "serious injury requiring emergency care, intense medical trauma, critical status",
        "highly disturbing injury photograph, close-up of severe physical trauma",
        "graphic aftermath of an accident, severe impact site with extensive physical injuries",
        "severe bodily trauma, catastrophic injury requiring immediate surgical intervention",
        "graphic disaster aftermath, severe structural destruction alongside casualties",
    ],

    "firearms": [
        "person holding a handgun, fingers near the trigger, close-up on a pistol weapon",
        "person holding a pistol, firearm extended, gripped with hands, tactical stance",
        "rifle visible in the image, long gun barrel, modern mechanical firearm structure",
        "shotgun leaning against a wall, barrel pointing upward, firearm in an interior setting",
        "firearm placed on a table, handgun or rifle resting on a flat surface, close-up shot",
        "armed individual, person carrying a noticeable weapon or firearm on their person",
        "person carrying a rifle, shoulder strap, long gun held across the torso",
        "close-up of a handgun, focus on the grip, trigger guard, and barrel of a pistol",
        "firearm displayed indoors, gun resting on a counter or mounted on a wall",
        "weapon visible in someone's hands, manual grip on a mechanical firearm device",
        "multiple firearms visible, collection of rifles and handguns arrayed together",
        "person aiming a firearm, extending arms, looking down the sight of a gun",
        "firearms inside a vehicle, gun resting on the passenger seat or dashboard",
        "person carrying a long gun, shotgun or rifle held by an individual outdoors",
        "firearm visible in a holster, pistol secured at the waist or belt of an individual",
    ],

    "bladed_weapons": [
        "person holding a knife, sharp metallic blade gripped in hand, forward stance",
        "knife visible on a table, steel blade resting on a surface, close-up frame",
        "large knife being displayed, long sharp edge, fixed blade weapon",
        "person carrying a machete, large utility blade held by an individual",
        "dagger visible, double-edged pointed blade weapon, historic or modern design",
        "sword being held, long metallic blade with hilt gripped by hands",
        "axe visible in the scene, metal hatchet head with a wooden or synthetic handle",
        "sharp blade in someone's hand, holding a metallic cutting weapon or tool",
        "kitchen knife prominently visible, chef knife or carving knife held outside a culinary context",
        "combat knife displayed, tactical fixed blade with textured grip handle",
        "bladed weapon on the ground, dropped knife or sword lying on a surface",
        "knife visible indoors, sharp cutting instrument resting on a household surface",
        "knife visible outdoors, blade held or placed in an outdoor environment",
        "large blade carried by a person, broad sword, machete, or cleaver held by hand",
        "sharp weapon clearly visible, focused shot on a pointed or edged metallic weapon",
    ],

    "explosives": [
        "grenade visible, fragmentation explosive shell resting on a surface or held",
        "explosive device, wired bundle, mechanism with components indicating detonation capability",
        "improvised explosive device, complex arrangement of wires, battery, and hazardous canisters",
        "bomb disposal scene, technician in a heavy protective suit inspecting an object",
        "explosive materials, commercial blasting caps, dynamite sticks, or military ordnances",
        "suspicious explosive object, package with visible external wiring and timers",
        "military explosive device, landmine, mortar shell, or missile casing visible",
        "explosion aftermath, smoke, debris cloud, and charred ruins from a blast",
        "detonation scene, sudden bright flash, shockwave, and scattering debris",
        "blast damage, shattered windows, collapsed walls, and smoke from an explosion",
    ],

    "drugs": [
        "person consuming illegal drugs, act of inhaling, injecting, or ingesting substances",
        "drug paraphernalia visible, pipes, specialized spoons, small digital scales, or foil",
        "illegal narcotics visible, bags of illicit powder, crystal substances, or pressed pills",
        "drug use in progress, context showing illicit substance consumption",
        "powdered illegal substance, white lines or piles on a glass or mirrored surface",
        "needle associated with drug use, hypodermic syringe next to illicit substances",
        "person preparing illegal drugs, crushing pills, heating substances, or dividing powder",
        "drug packaging visible, small clear plastic ziplock baggies containing substances",
        "drug-related objects, rolled currency, bongs, or cut straws on a table surface",
        "illegal substance on a table, unregulated chemical compounds prepared for use",
        "person handling narcotics, transferring packets of illicit materials by hand",
        "drug consumption indoors, dark room environment with substance use occurring",
        "drug consumption outdoors, hidden or secluded public space with substance use",
        "controlled substance visible, illicit unprescribed narcotics grouped together",
        "drug paraphernalia on a surface, equipment used for illegal substance consumption",
    ],

    "alcohol": [
        "person drinking beer, tilting a glass bottle or aluminum can toward their mouth",
        "person consuming wine, holding a stemmed glass filled with red or white liquid",
        "person holding a liquor bottle, glass bottle with an alcohol label held by the neck",
        "bar counter with alcoholic drinks, rows of liquor bottles, draft taps, and filled glasses",
        "cocktail in a person's hand, mixed drink with ice and a garnish held at a social gathering",
        "beer bottles on a table, multiple amber glass bottles clustered together indoors or outdoors",
        "wine glasses during dinner, elegant table setting with partially filled stemware",
        "group drinking alcohol, people raising glasses in a toast, social drinking environment",
        "person pouring alcohol, liquid transferring from a bottle into a shot or cocktail glass",
        "alcoholic beverages at a party, coolers filled with cans, drinks arranged on a table",
    ],

    "tobacco_vaping": [
        "person smoking a cigarette, holding a lit paper cylinder between fingers with smoke rising",
        "person smoking a cigar, thick rolled tobacco product held to lips, dense smoke plume",
        "person using an electronic cigarette, hand-held vape mod or pod system held to the mouth",
        "person vaping, inhaling from a sleek metallic electronic nicotine delivery device",
        "vape device visible, electronic cigarette resting on a flat surface, close-up shot",
        "smoke from a cigarette, thin white vapor trails emanating from a burning tip",
        "person holding a lit cigarette, ember visible at the tip, held casually between fingers",
        "tobacco products on a table, loose loose-leaf packs, loose cigarettes, or lighters",
        "person exhaling vape aerosol, thick white cloud of vapor coming from a person's mouth",
        "close-up of a vaping device, details of the tank, mouthpiece, and battery mod unit",
    ],

    "self_harm": [
        "person appearing to engage in self-harm, desperate gestures, dangerous isolation context",
        "scene suggesting self-inflicted injury, highly distressing scenario with sharp elements or risks",
        "individual in severe emotional distress, hands over face, weeping, intense crisis posture",
        "medical emergency related to self-injury, emergency responder intervention context",
        "person requiring immediate assistance, acute emotional collapse or physical jeopardy",
        "evidence of possible self-harm, distressing individual crisis situation shown visually",
        "distressing scene involving one individual, extreme isolation, profound psychological despair",
        "individual in crisis, dangerous or high-risk behavior directed at oneself",
        "scene requiring emergency intervention, critical personal crisis or self-directed danger",
        "person at risk of serious harm, critical safety risk involving an individual alone",
    ],

    "fire_disaster": [
        "building on fire, large structure engulfed in flames, massive black smoke plumes",
        "house fire, residential home burning, bright orange flames breakthrough windows",
        "forest fire, trees burning uncontrollably, woodland landscape consumed by wildfire",
        "vehicle engulfed in flames, car or truck burning fiercely on the side of a road",
        "wildfire approaching homes, wall of fire near a residential neighborhood, smoke-filled sky",
        "collapsed building, piles of concrete rubble, exposed rebar, structural failure site",
        "earthquake damage, cracked asphalt streets, buckled roads, and ruined buildings",
        "flooded street, muddy water filling an urban area up to car windows, natural disaster",
        "major storm damage, uprooted trees, downed power lines, and ripped roofs after a hurricane",
        "natural disaster aftermath, widespread environmental destruction, flooded or ruined landscape",
        "rescue workers during a disaster, emergency personnel navigating debris or floodwaters",
        "firefighters responding to a fire, fire truck parked, crew deploying hoses toward a burning structure",
    ],

    "hate_symbols": [
        "hate symbol displayed, explicit graphic icon associated with hate groups or extremism",
        "extremist symbol visible, radical political emblem or offensive historical marker",
        "symbol associated with hate groups, discriminatory branding or white supremacist insignia",
        "racist symbol, historical or modern emblem used to promote racial supremacy",
        "extremist flag, banner displaying an offensive or banned radical group symbol",
        "offensive hate imagery, visual iconography meant to demean or intimidate protected groups",
        "hate-related graffiti, offensive symbols sprayed onto a public wall or barrier",
        "extremist insignia, uniform patches or markings representing radical ideology",
        "symbol promoting hatred, visually clear discriminatory or malicious emblem",
        "public display of hate symbols, offensive banners or markings in a public area",
    ],

    "offensive_gesture": [
        "person making an obscene hand gesture, explicit vulgar finger sign directed at the camera",
        "offensive gesture directed at another person, aggressive hand signal during a dispute",
        "aggressive insulting gesture, hostile physical sign meant to demean or provoke",
        "hostile hand gesture, crude manual signal indicating intense anger or disrespect",
        "person making a rude gesture, showing disrespect or contempt with a hand sign",
        "gesture intended to insult, explicit and vulgar physical movement or posture",
        "offensive body language, hostile framing with crude expressions and insulting hand motions",
        "public display of an obscene gesture, explicit vulgar hand sign captured clearly in public",
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
            image_embeds = outputs.image_embeds[0].tolist()
            return ClipResult(available=True, scores=scores, model_name=CLIP_MODEL_NAME, image_embedding=image_embeds)
        except Exception as exc:
            return ClipResult(available=False, model_name=CLIP_MODEL_NAME, error=str(exc))

    def _load(self):
        if self._processor is not None:
            return self._processor, self._model
        if self._load_error is not None:
            return None, None
        try:
            from transformers import CLIPModel, CLIPProcessor
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
    """Aggregate prompt probabilities into cluster scores and keep only top matches."""
    scores: dict[str, float] = {}
    for cluster_id, prob in zip(prompt_cluster_ids, prompt_probs):
        scores[cluster_id] = scores.get(cluster_id, 0.0) + float(prob)

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
