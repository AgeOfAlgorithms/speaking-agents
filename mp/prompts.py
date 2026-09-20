"""How the multiplayer game is put to Laya: the instruction, and one short text option per action.

Laya scores options written as text, so the option list can change every step. We use that:
  * only actions that would work right now are offered (mp.engine.MPEnv.available), and
  * `use` is described by what it would do on the tile you face ("use: chop the tree").
Directions are compass words, the same ones the text state and the spoken language use.
"""
from crafter import constants

from mp.engine import ACTIONS, BUCKETS, CAMPFIRE

INSTRUCTIONS = (
    "You are {name}, playing a cooperative survival game with teammates. The team scores once for each achievement "
    "that anyone unlocks: collect wood, stone, coal, iron, diamond, sapling and water; craft pickaxes and swords; place "
    "a table, furnace, plant and stone; eat, sleep, defeat monsters. So divide the work, share items, stay alive, and "
    "revive teammates who are down. Choose your next action."
)

LABEL = {"noop": "wait", "move_left": "go_west", "move_right": "go_east", "move_up": "go_north", "move_down": "go_south", "do": "use"}
PURPOSE = {"wood_pickaxe": "mines stone and coal", "stone_pickaxe": "mines iron", "iron_pickaxe": "mines diamond",
           "wood_sword": "double damage", "stone_sword": "triple damage", "iron_sword": "five times damage",
           "wood_bucket": "carries %d drinks" % BUCKETS["wood_bucket"], "iron_bucket": "carries %d drinks" % BUCKETS["iron_bucket"]}


def _cost(uses):
    return ", ".join("%d %s" % (v, k) for k, v in uses.items())


def describe(action, use_label="use the tile you face"):
    if action == "noop":
        return "do nothing this turn"
    if action.startswith("move_"):
        return "walk one tile %s, or turn to face it if blocked" % LABEL[action][3:]
    if action == "do":
        return use_label
    if action == "sleep":
        return "sleep to restore energy; unsafe if monsters are near"
    if action.startswith("make_"):
        k = action[5:]
        return "craft it (%s); %s" % (_cost(constants.make[k]["uses"]), PURPOSE[k])
    if action in ("place_campfire", "place_stone_campfire"):
        spec = CAMPFIRE["kinds"][action[6:]]
        lasts = "the whole night" if spec["burn"] >= CAMPFIRE["night_steps"] else "a third of the night"
        return "light it ahead (%s); zombies and shades cannot come near, skeletons can; burns %s, feed it wood" % (_cost(spec["cost"]), lasts)
    if action == "place_bucket":
        return "set your bucket down so teammates can drink from it"
    if action == "drink_bucket":
        return "drink from the bucket you carry; fully quenches thirst"
    if action.startswith("place_"):
        k = action[6:]
        why = {"stone": "blocks monsters, bridges water", "table": "needed nearby to craft", "furnace": "needed for iron tools",
               "plant": "grows into food"}[k]
        return "put it on the tile ahead (%s); %s" % (_cost(constants.place[k]["uses"]), why)
    if action.startswith("drop_"):
        return "drop one on the tile ahead so a teammate can pick it up"
    raise KeyError(action)


def label(action):
    return LABEL.get(action, action)


SPEAK = "speak"
SPEAK_TEXT = "say one short sentence to teammates within earshot instead of acting this turn"


def question_from_offered(name, offered, use_label):
    """The same question as action_question, rebuilt from a recorded option list (training data)."""
    criteria = {label(a): (SPEAK_TEXT if a == SPEAK else describe(a, use_label)) for a in offered}
    return {"type": "choice", "instructions": INSTRUCTIONS.format(name=name), "criteria": criteria}


def action_question(P, mask=True, can_speak=True):
    """-> (question dict, [engine action names in option order]). The last option may be SPEAK: saying
    something is a move like any other, and it uses up the turn."""
    blocked = P["blocked"] if mask else {}
    offered = [a for a in ACTIONS if a not in blocked] or ["noop"]
    criteria = {label(a): describe(a, P["use"][0]) for a in offered}
    if can_speak:
        offered = offered + [SPEAK]
        criteria[SPEAK] = SPEAK_TEXT
    return {"type": "choice", "instructions": INSTRUCTIONS.format(name=P["name"]), "criteria": criteria}, offered
