"""Controllers for the live multiplayer game. Each returns (action_name, speech_or_None) per step.
Speaking takes the turn: when speech is returned, the engine ignores the action.

ScriptedAgent is the hand-written cooperative baseline ("what would talking buy you if the protocol
were given for free?"): it plays with the single-player teacher's rules and adds a fixed protocol --
announce useful finds, call for help when in trouble, go to a teammate who calls, revive the downed.
"""
import numpy as np

from core import CX, CY, DIRS
from mp import language as L
from mp.engine import ACTIONS
from teacher import Teacher

WATER_STATION = ("placed_wood_bucket", "placed_iron_bucket")          # a set-down bucket that still has water
ANY_BUCKET = WATER_STATION + ("placed_wood_bucket_empty", "placed_iron_bucket_empty")
LIT = ("campfire", "campfire_low", "stone_campfire", "stone_campfire_low")
OUT = ("campfire_out", "stone_campfire_out")


class RandomAgent:
    kind = "random"

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def act(self, P):
        speech = tuple(int(self.rng.randint(k)) for k in L.SIZES) if self.rng.rand() < 0.03 else None
        return ACTIONS[self.rng.randint(len(ACTIONS))], speech


class ScriptedAgent:
    """Hand-written cooperative player. Crafter itself is played by the single-player rules in teacher.py;
    on top of that sit the team behaviours, each chosen so that one sentence is worth the turn it costs:

      night camp   at dusk, light a campfire (3 wood) and call the team to it; everyone sleeps inside the
                   firelight, where zombies cannot reach. Without the call, each player needs a fire of
                   their own and whoever has no wood spends the night in the open.
      finds        tell teammates who are out of sight where water / coal / iron / diamond is. A listener
                   adds it to what they "remember", so they can walk there instead of searching.
      rescue       a downed player calls for help; a healthy teammate comes and revives.
    The silent variant behaves identically but never speaks (and so never benefits from being told)."""
    kind = "scripted"
    FINDS = ("water", "coal", "iron", "diamond")

    def __init__(self, seed=0, talk=True):
        self.teacher = Teacher(seed)
        self.talk = talk
        self.announced = {}        # thing -> step last announced
        self.told = {}             # thing -> global position a teammate told us about
        self.rescue = None         # [global x, global y, steps left] of a teammate who called for help
        self.camp = None           # global position of the campfire we use
        self.last_call = self.camp_called = self.camp_seen = -999
        self.deferred = -1
        self.stone_seen = -999
        self.asked = -999          # when we last asked for something
        self.requests = {}         # item -> (step heard, requester's global position)
        self.giving = None         # [item, how many still to drop, requester position, when they asked]

    def act(self, P):
        step, pos, inv, g = P["step"], P["pos"], P["inv"], P["grid"]
        if P["downed"]:
            if self.talk and step - self.last_call >= 8:
                self.last_call = step
                return "noop", L.encode(verb="help", object="me")
            return "noop", None

        for h in P["heard"]:                                   # listening costs nothing
            sx, sy = self._parse(h["where"])
            speaker = (pos[0] + sx, pos[1] + sy)
            if "help" in h["text"] and inv["health"] >= 5:
                self.rescue = [speaker[0], speaker[1], 60]
            elif "need" in h["text"].split():
                for item in ("wood", "stone"):
                    if item in h["text"].split():
                        self.requests[item] = (step, speaker)
            elif "campfire" in h["text"]:
                self.camp, self.camp_seen = speaker, step      # a fresh call: that fire is burning now
            else:
                off = L.offset_from_words(h["text"])
                thing = next((t for t in self.FINDS if t in h["text"].split()), None)
                if off and thing:
                    self.told[thing] = (speaker[0] + off[0], speaker[1] + off[1])

        cells = lambda *names: [(x, y) for x in range(len(g)) for y in range(len(g[0])) if g[x][y] in names]
        hostile_adjacent = any(g[CX + dx][CY + dy] in ("zombie", "skeleton", "shade") for dx, dy in DIRS.values())
        if hostile_adjacent:
            return ACTIONS[self.teacher.act(P)], None           # fight first; talking would cost the turn

        downed = [t for t in P["team"] if t["downed"]]
        if downed:                                              # revive anyone we can see
            ox, oy = downed[0]["offset"]
            r = self.teacher.approach(P, {(CX + ox, CY + oy)})
            if r == "READY":
                self.rescue = None
                return "do", None
            if r:
                return r, None
        if self.rescue:
            self.rescue[2] -= 1
            off = (self.rescue[0] - pos[0], self.rescue[1] - pos[1])
            if self.rescue[2] <= 0 or (abs(off[0]) <= 1 and abs(off[1]) <= 1):
                self.rescue = None
            else:
                return self.teacher.goto_offset(P, off), None

        light, falling = P["daylight"], P["light_trend"] == "falling"
        starving = min(inv["food"], inv["drink"]) <= 2             # hunger and thirst outrank staying in camp
        r = self._share(P, cells, light, falling)
        if r:
            return r
        carried = sum(P["bucket_water"].values())
        has_bucket = inv["wood_bucket"] or inv["iron_bucket"]

        # water for the night: the drink bar lasts ~180 waking steps and the night is ~196, so a camp needs
        # a bucket. Drink from one whenever thirsty, anywhere.
        if inv["drink"] <= 5:
            if carried and "drink_bucket" not in P["blocked"]:
                return "drink_bucket", None
            stations = cells(*WATER_STATION)
            if stations:
                r = self.teacher.approach(P, set(stations))
                if r:
                    return ("do" if r == "READY" else r), None
        if light >= 0.85:
            if not has_bucket and not cells(*ANY_BUCKET) and "make_wood_bucket" not in P["blocked"] and inv["wood_pickaxe"]:
                return "make_wood_bucket", None
            if not has_bucket and cells(*ANY_BUCKET) and not (falling and light < 0.99):
                r = self.teacher.approach(P, set(cells(*ANY_BUCKET)))   # morning: take the camp's bucket along again
                if r and (r != "READY" or inv["drink"] >= 9 or cells("placed_wood_bucket_empty", "placed_iron_bucket_empty")):
                    return ("do" if r == "READY" else r), None
            if has_bucket and carried < 3 and cells("water"):            # top the bucket up whenever water is on screen
                r = self.teacher.approach(P, set(cells("water")))
                if r:
                    return ("do" if r == "READY" else r), None
        # head for the fire in the afternoon (light starts falling ~90 steps before the zombies come out):
        # most players who went down at night were still walking to a camp that was too far away
        if light < 0.85 and not starving:                           # zombies are about: be at a fire
            r = self._night(P, cells)
            if r:
                return r
        elif falling and light < 0.99 and not starving:              # afternoon: stock up, tonight's fire eats wood
            if inv["wood"] < 7 and cells("tree"):
                r = self._fetch_wood(P, cells)
                if r:
                    return r
            if self.camp and step - self.camp_seen <= 70:            # someone already has a fire going: join them
                ox, oy = self.camp[0] - pos[0], self.camp[1] - pos[1]
                if max(abs(ox), abs(oy)) > 2:
                    return self.teacher.goto_offset(P, (ox, oy)), None

        # day: tell teammates who cannot see it where the useful things are
        if self.talk and not cells("zombie", "skeleton", "shade") and len(P["team"]) < len(P["teammates"]):
            for thing in self.FINDS:
                spots = cells(thing)
                if spots and step - self.announced.get(thing, -999) >= 300:
                    x, y = min(spots, key=lambda c: abs(c[0] - CX) + abs(c[1] - CY))
                    self.announced[thing] = step
                    return "noop", L.encode(subject="I", verb="see", object=thing, direction=L.direction_of(x - CX, y - CY),
                                            distance=L.distance_of(x - CX, y - CY))

        # what we were told becomes part of what we remember, unless we know better ourselves
        told = {t: (gp[0] - pos[0], gp[1] - pos[1]) for t, gp in self.told.items()
                if t not in P["remembered"] and not cells(t)}
        for t in list(self.told):
            gx, gy = self.told[t][0] - pos[0], self.told[t][1] - pos[1]
            if abs(gx) <= 2 and abs(gy) <= 2:                   # arrived: whatever is here, we see it now
                del self.told[t]
        if told:
            P = dict(P, remembered={**P["remembered"], **told})
        if inv["wood_pickaxe"]:                                      # once the basics exist, reserve 3 stone + 4 wood for the fire
            P = dict(P, inv={**inv, "stone": max(0, inv["stone"] - 3), "wood": max(0, inv["wood"] - (4 if has_bucket else 6))})
        return ACTIONS[self.teacher.act(P)], None

    def _night(self, P, cells):
        """Evening and night. Fires are short now (a wood fire lasts a third of the night), and a sleeper
        cannot feed one, so a camp needs wood in stock and one player awake on watch."""
        pos, inv, g, step = P["pos"], P["inv"], P["grid"], P["step"]
        near = lambda spots: min(spots, key=lambda c: abs(c[0] - CX) + abs(c[1] - CY))
        lit = cells(*LIT)
        if lit:
            x, y = near(lit)
            self.camp, self.camp_seen = (pos[0] + x - CX, pos[1] + y - CY), step
        if self.camp and step - self.camp_seen > 70:
            self.camp = None                                    # not seen burning for a while: assume it went out

        dying = cells("campfire_low", "stone_campfire_low") or ([] if lit else cells(*OUT))
        if dying and inv["wood"] > 0:                           # feed a dying fire, or relight a dead one
            r = self.teacher.approach(P, {near(dying)})
            if r:
                return ("do" if r == "READY" else r), None

        if self.camp:
            ox, oy = self.camp[0] - pos[0], self.camp[1] - pos[1]
            dist = max(abs(ox), abs(oy))
            if dist <= 2 and lit:
                if self.talk and step - self.camp_called >= 60 and len(P["team"]) < len(P["teammates"]):
                    self.camp_called = step                     # someone is still out there: call them in
                    return "noop", L.encode(verb="come", object="campfire", direction="here")
                if sum(P["bucket_water"].values()) and "place_bucket" not in P["blocked"] and not cells(*WATER_STATION):
                    return "place_bucket", None                 # water for everyone, all night
                awake = [t["name"] for t in P["team"] if not t["asleep"] and not t["downed"]]
                on_watch = not awake or P["name"] < min(awake)  # one player, the same one everybody would pick
                if on_watch and inv["energy"] > 2:
                    return "noop", None                         # stay awake so the fire can be fed
                return ("sleep" if inv["energy"] < 9 else "noop"), None
            if dist <= 2 and not lit:
                self.camp = None                                # we are where the fire should be and it is not burning
            elif dist <= 12 or inv["wood"] < 3:
                return self.teacher.goto_offset(P, (ox, oy)), None
            else:
                self.camp = None                                # too far to reach before dark: build one here

        if inv["wood"] >= 3:
            neighbours = [t for t in P["team"] if not t["downed"] and max(abs(t["offset"][0]), abs(t["offset"][1])) <= 5]
            if any(t["name"] < P["name"] for t in neighbours):   # a teammate right here will light one: give them a moment
                self.deferred = step if self.deferred < 0 else self.deferred
                if step - self.deferred < 4:
                    return "noop", None
            else:
                self.deferred = -1
            build ="place_stone_campfire" if inv["stone"] >= 3 else "place_campfire"   # stone lasts the whole night
            if build not in P["blocked"]:
                f = P["facing"]
                self.camp, self.camp_seen, self.camp_called = (pos[0] + f[0], pos[1] + f[1]), step, -999
                return build, None
            for d, (dx, dy) in DIRS.items():                    # step somewhere with room to build
                if g[CX + dx][CY + dy] in ("grass", "sand", "path") and g[CX + 2 * dx][CY + 2 * dy] in ("grass", "sand", "path"):
                    return "move_" + d, None
            return None
        return self._fetch_wood(P, cells)

    def _share(self, P, cells, light, falling):
        """A stone campfire needs 3 wood AND 3 stone in one pair of hands, and inventories are private. So:
        say what you lack, walk to whoever asked and drop what you can spare, pick up what was dropped for you."""
        pos, inv, step = P["pos"], P["inv"], P["step"]
        evening = (falling and light < 0.99) or light < 0.85
        lacking = [i for i in ("wood", "stone") if inv[i] < 3]
        want = [i for i in lacking if cells("dropped_" + i)] if evening else []
        if want:                                                # something we need is lying right there
            r = self.teacher.approach(P, set(cells("dropped_" + want[0])))
            if r:
                return ("do" if r == "READY" else r), None

        if self.giving:
            item, left, where, heard = self.giving
            ox, oy = where[0] - pos[0], where[1] - pos[1]
            if left <= 0 or inv[item] < 1 or step - heard > 60:
                self.giving = None
            elif max(abs(ox), abs(oy)) > 2:
                return self.teacher.goto_offset(P, (ox, oy)), None
            elif "drop_" + item not in P["blocked"]:
                self.giving[1] -= 1
                return "drop_" + item, None
            else:
                return self.teacher.explore(P), None             # no free tile in front: shuffle and try again
        for item, (heard, where) in list(self.requests.items()):
            fresh = step - heard <= 25
            spare = inv[item] >= 6 or (inv[item] >= 3 and any(inv[o] < 3 for o in ("wood", "stone") if o != item))
            near = max(abs(where[0] - pos[0]), abs(where[1] - pos[1])) <= 12
            if fresh and spare and near and not self.giving:
                self.giving = [item, 3, where, heard]
                del self.requests[item]
                return self._share(P, cells, light, falling)
            if not fresh:
                del self.requests[item]

        if cells("stone_campfire", "stone_campfire_low"):
            self.stone_seen = step
        settled = step - self.stone_seen <= 70                  # a stone fire is burning: tonight is taken care of
        if (self.talk and evening and not settled and len(lacking) == 1 and P["teammates"]
                and step - self.asked >= 40 and not cells(*["dropped_" + i for i in lacking])):
            self.asked = step                                   # I hold one half of a stone campfire: ask for the other
            return "noop", L.encode(verb="need", object=lacking[0])
        return None

    def _fetch_wood(self, P, cells):
        trees = cells("tree")
        if trees:
            r = self.teacher.use(P, set(trees))
            if r:
                return r, None
        if "tree" in P["remembered"]:
            return self.teacher.goto_offset(P, P["remembered"]["tree"]), None
        return None

    @staticmethod
    def _parse(where):
        dx = dy = 0
        parts = where.split()
        for n, d in zip(parts[::2], parts[1::2]):
            n = int(n)
            dx += n if d == "east" else -n if d == "west" else 0
            dy += n if d == "south" else -n if d == "north" else 0
        return dx, dy


class HumanAgent:
    kind = "human"

    def __init__(self):
        self.action, self.speech = None, None

    def act(self, P):
        a, s = self.action or "noop", self.speech
        self.action = self.speech = None
        return a, s


def make(kind, seed=0):
    if kind == "human":
        return HumanAgent()
    if kind == "random":
        return RandomAgent(seed)
    if kind == "laya_zeroshot":
        from mp.laya_policy import LayaAgent   # imported on demand: pulls in torch and the 421M model
        return LayaAgent("convaiinnovations/laya", kind=kind)
    if kind == "laya_trained":
        from mp.laya_policy import LayaAgent, TrainedMP
        return LayaAgent("models/team_bc", kind=kind, backend=TrainedMP)
    if kind == "von_zeroshot":
        from mp.laya_policy import LayaAgent, VonMP
        return LayaAgent("wfzyx/von-1.0", kind=kind, backend=VonMP)
    if kind == "scripted_silent":
        return ScriptedAgent(seed, talk=False)
    return ScriptedAgent(seed)
