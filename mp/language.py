"""The agents' spoken language: six slots, each filled from a small closed vocabulary or left blank.

    subject   modal    verb    object   direction    distance
    "I"       "will"   "chop"  "tree"   "northwest"  "pretty close"   ->  "I will chop tree northwest pretty close"
    ""        ""       "need"  "wood"   ""           ""               ->  "need wood"

Every slot is a `choice` question for Laya (blank is just another option), so a sentence costs no
text generation. The words are ordinary English on purpose: a pretrained language encoder already
has meanings for them, which is the thing being tested against a shuffled vocabulary.

North is up on the screen. `direction_of` / `distance_of` define what the place words mean in tiles,
so scripted speakers, the text state and any evaluation of "was that true?" all agree.
"""
SLOTS = ("subject", "modal", "verb", "object", "direction", "distance")

VOCAB = {
    "subject": ["", "I", "you", "we", "he", "she", "they", "zombie", "skeleton", "shade", "cow"],
    "modal": ["", "will", "should", "must", "might", "could", "can", "cannot", "did"],
    "verb": ["", "go", "come", "run", "follow", "wait", "help", "attack", "eat", "drink", "sleep", "chop", "mine",
             "craft", "place", "drop", "take", "give", "need", "have", "see"],
    "object": ["", "tree", "wood", "stone", "coal", "iron", "diamond", "water", "cow", "food", "zombie", "skeleton", "shade",
               "table", "furnace", "campfire", "bucket", "pickaxe", "sword", "sapling", "lava", "me", "you", "it"],
    "direction": ["", "here", "north", "south", "east", "west", "northeast", "northwest", "southeast", "southwest"],
    "distance": ["", "close by", "pretty close", "pretty far", "far away", "very far"],
}
DISTANCE_BANDS = (("close by", 3), ("pretty close", 7), ("pretty far", 15), ("far away", 30), ("very far", 10 ** 9))


def direction_of(dx, dy):
    """Offset in tiles (x grows east, y grows south) -> compass word. Diagonal unless one axis dominates 2:1."""
    if dx == 0 and dy == 0:
        return "here"
    ns = "north" if dy < 0 else "south"
    ew = "east" if dx > 0 else "west"
    if abs(dx) >= 2 * abs(dy):
        return ew
    if abs(dy) >= 2 * abs(dx):
        return ns
    return ns + ew


UNIT = {"here": (0, 0), "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
        "northeast": (1, -1), "northwest": (-1, -1), "southeast": (1, 1), "southwest": (-1, 1)}
BAND_MIDDLE = {"close by": 2, "pretty close": 5, "pretty far": 11, "far away": 22, "very far": 40}


def offset_from_words(text):
    """The listener's best guess at "<direction> <distance>" in a heard sentence -> (dx, dy), or None."""
    direction = next((d for d in sorted(UNIT, key=len, reverse=True) if d in text.split()), None)
    if direction is None:
        return None
    dist = next((BAND_MIDDLE[b] for b in BAND_MIDDLE if b in text), 4)
    ux, uy = UNIT[direction]
    return ux * dist, uy * dist


def distance_of(dx, dy):
    d = max(abs(dx), abs(dy))
    return next(word for word, limit in DISTANCE_BANDS if d <= limit)


SIZES = tuple(len(VOCAB[s]) for s in SLOTS)
SILENT = (0,) * len(SLOTS)


def render(ids):
    """Slot indices -> "I see water northwest pretty close" ('' if every slot is blank)."""
    return " ".join(w for w in (VOCAB[s][i] for s, i in zip(SLOTS, ids)) if w)


def encode(**words):
    """encode(subject="I", verb="see", object="water", direction="northwest", distance="pretty close") -> slot indices."""
    return tuple(VOCAB[s].index(words.get(s, "")) for s in SLOTS)


def is_silent(ids):
    return ids is None or not any(ids)
