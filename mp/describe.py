"""True things a player could say about the state in front of it -- computed from the text state alone.

Used to teach the speech decoder to SPEAK before anyone has to learn to LISTEN (mp/pretrain_speech.py):
for any recorded game state this gives sentences that are true of it, in the agents' vocabulary, so the
decoder learns how words attach to the world (which compass word, which distance, how many, who) from
as many examples as we like and without running the game. When saying something is worth a turn, and
what a listener should do about it, is left to imitation and reinforcement learning.

    true_sentences(state, offered) -> {family: [sentence, ...]}      sentence = list of words
"""
import json
import re

from mp import language
from mp.model import MAX_WORDS, WORD_ID

# how the state's object names are said
SAY = {"tree": "tree", "cow": "cow", "water": "water", "table": "table", "furnace": "furnace", "sapling": "sapling", "stone": "stone",
       "coal": "coal", "iron": "iron", "diamond": "diamond", "lava": "lava", "zombie": "zombie", "skeleton": "skeleton", "shade": "shade",
       "campfire": "campfire", "campfire_low": "campfire", "stone_campfire": "campfire", "stone_campfire_low": "campfire",
       "placed_wood_bucket": "bucket", "placed_iron_bucket": "bucket", "dropped_wood": "wood", "dropped_stone": "stone",
       "dropped_coal": "coal", "dropped_iron": "iron", "dropped_diamond": "diamond", "dropped_sapling": "sapling"}
MONSTERS = ("zombie", "skeleton", "shade")
HELD = {"wood": "wood", "stone": "stone", "coal": "coal", "iron": "iron", "diamond": "diamond", "sapling": "sapling",
        "wood_pickaxe": "pickaxe", "stone_pickaxe": "pickaxe", "iron_pickaxe": "pickaxe", "wood_sword": "sword", "stone_sword": "sword",
        "iron_sword": "sword", "wood_bucket": "bucket", "iron_bucket": "bucket"}
MAP_COUNT = {"T": "tree", "O": "cow", "Z": "zombie", "K": "skeleton", "V": "shade"}
DID = {"collect wood": "I did chop tree", "collect stone": "I did mine stone", "collect coal": "I did mine coal", "collect iron": "I did mine iron",
       "collect diamond": "I did mine diamond", "collect drink": "I did drink water", "collect sapling": "I did take sapling",
       "eat cow": "I did eat cow", "place table": "I did place table", "place furnace": "I did place furnace", "place plant": "I did place sapling",
       "place stone": "I did place stone", "place campfire": "I did place campfire", "place stone campfire": "I did place campfire",
       "make wood pickaxe": "I did make pickaxe", "make stone pickaxe": "I did make pickaxe", "make iron pickaxe": "I did make pickaxe",
       "make wood sword": "I did make sword", "make stone sword": "I did make sword", "make iron sword": "I did make sword",
       "make wood bucket": "I did make bucket", "make iron bucket": "I did make bucket", "defeat zombie": "I did attack zombie",
       "defeat skeleton": "I did attack skeleton", "defeat shade": "I did attack shade", "woke up": "I did sleep", "wake up": "I did sleep"}
CAN = {"make_wood_pickaxe": "I can make pickaxe", "make_stone_pickaxe": "I can make pickaxe", "make_iron_pickaxe": "I can make pickaxe",
       "make_wood_sword": "I can make sword", "make_stone_sword": "I can make sword", "make_iron_sword": "I can make sword",
       "make_wood_bucket": "I can make bucket", "make_iron_bucket": "I can make bucket", "place_table": "I can place table",
       "place_furnace": "I can place furnace", "place_campfire": "I can place campfire", "place_stone_campfire": "I can place campfire",
       "place_bucket": "I can place bucket"}


def offset(text):
    """'4 west 3 south' -> (-4, 3); 'here' -> (0, 0)."""
    dx = dy = 0
    for n, d in re.findall(r"(\d+) (west|east|north|south)", text):
        n = int(n)
        dx += {"west": -n, "east": n}.get(d, 0)
        dy += {"north": -n, "south": n}.get(d, 0)
    return dx, dy


def where(text):
    dx, dy = offset(text)
    d = language.direction_of(dx, dy)
    return [d] if d == "here" else [d] + language.distance_of(dx, dy).split()


def amount(n):
    """Ways of saying how many: 1 -> nothing; 2-3 -> few / multiple; 4+ -> many / multiple."""
    return [[]] if n <= 1 else [["few"], ["multiple"]] if n <= 3 else [["many"], ["multiple"]]


def true_sentences(state, offered=()):
    s = json.loads(state) if isinstance(state, str) else state
    out = {}

    def add(family, words):
        words = words.split() if isinstance(words, str) else words
        if len(words) <= MAX_WORDS and all(w in WORD_ID or w in names for w in words):
            out.setdefault(family, []).append(words)

    names = [s["you"]] + list(s.get("teammates", []))
    counts = {}
    for row in s.get("map", []):
        for ch in row.split():
            if ch in MAP_COUNT:
                counts[MAP_COUNT[ch]] = counts.get(MAP_COUNT[ch], 0) + 1
    for obj, pos in s.get("nearest", {}).items():
        word = SAY.get(obj)
        if word is None:
            continue
        if word in MONSTERS:
            for q in amount(counts.get(word, 1)):
                add("danger", q + [word] + where(pos))
                add("danger", ["danger"] + q + [word] + where(pos)[:1])
        else:
            for q in amount(counts.get(word, 1)):
                add("see", ["I", "see"] + q + [word] + where(pos))
    for obj, pos in s.get("remembered", {}).items():
        if obj in SAY and obj not in s.get("nearest", {}):
            add("remember", [SAY[obj], "is"] + where(pos))
    held = {}
    for item, n in s.get("inventory", {}).items():
        if item in HELD:
            held[HELD[item]] = held.get(HELD[item], 0) + n
    for word, n in held.items():
        for q in amount(n if word in ("wood", "stone", "coal", "iron", "diamond", "sapling") else 1):
            add("have", ["I", "have"] + q + [word])
    for word in ("wood", "stone", "pickaxe", "sword"):
        if word not in held:
            add("have_not", ["I", "have", "no", word])
    if s.get("bucket_water") and any(s["bucket_water"].values()):
        add("have", "I have water bucket")
    if s.get("downed"):
        add("need", "help me")
        add("need", "I need help")
    else:
        if s["food"] <= 4:
            add("need", "I need food")
        if s["drink"] <= 4:
            add("need", "I need water")
        if s["energy"] <= 3:
            add("need", "I need sleep")
        if s["health"] <= 4:
            add("need", "I need help")
    pct, trend = re.match(r"(\d+)% ?(\w*)", s.get("light", "100%")).groups()
    pct = int(pct)
    if s.get("time") == "night":
        add("time", "it is night")
        add("time", "it is night now")
        if trend == "rising" and pct >= 40:
            add("time", "day soon")
    elif s.get("time") == "day":
        add("time", "it is day")
        if trend == "falling" and pct < 97:
            add("time", "night soon")
    else:
        add("time", "night soon" if trend == "falling" else "day soon")
    for name, pos in s.get("teammates_in_view", {}).items():
        add("teammate", [name, "is"] + where(pos))
        if "downed" in pos:
            add("teammate", ["help", name] + where(pos)[:1])
            add("teammate", [name, "need", "help"])
    if s.get("table_nearby"):
        add("here", "table is here")
    if s.get("furnace_nearby"):
        add("here", "furnace is here")
    for e in s.get("recent_events", []):
        m = re.match(r"(\d+) ago: ([a-z ]+?)(?: x\d+)?$", e)
        if m and int(m.group(1)) <= 5 and m.group(2) in DID:
            add("did", DID[m.group(2)])
    for a in offered:
        if a in CAN:
            add("can", CAN[a])
    return out


def all_true(state, offered=()):
    return {" ".join(w) for ws in true_sentences(state, offered).values() for w in ws}


if __name__ == "__main__":                      # python -m mp.describe : a few examples from the recorded data
    import itertools
    import random
    random.seed(1)
    with open("data/warmstart/samples.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in itertools.islice(f, 0, 60000, 4001)]
    fams, n = {}, 0
    for r in rows:
        t = true_sentences(r["state"], r["offered"])
        n += sum(len(v) for v in t.values())
        print(json.loads(r["state"])["you"], "->", " | ".join(" ".join(random.choice(v)) for v in t.values()))
    print("%.1f true sentences per state" % (n / len(rows)))
