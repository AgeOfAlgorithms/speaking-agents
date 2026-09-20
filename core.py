"""What a player perceives, for Multiplayer Crafter (mp/).

Observer   -- per-player wrapper that turns the world into a `percept` (the 9x7 local view, stats,
              inventory) plus two kinds of memory: a short-term event log and remembered landmarks.
state_dict -- the percept as the JSON-able text state an agent reads. mp.engine.state_dict adds the
              social parts (teammates, buckets, what was heard).

A percept only contains what is visible on screen (local view + inventory bar) or was visible or
audible earlier, so scripted and learned agents are told exactly the same things.
"""
import json

import numpy as np
from crafter import constants, objects

ACTIONS = list(constants.actions)
A = {n: i for i, n in enumerate(ACTIONS)}
ACHIEVEMENTS = list(constants.achievements)
VIEW_W, VIEW_H = 9, 7
CX, CY = 4, 3
DIRS = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}
DIR_NAME = {v: k for k, v in DIRS.items()}
WALKABLE = set(constants.walkable)
STATS = ("health", "food", "drink", "energy")
TOOLS = ("wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword")

# what is worth naming in "nearest" / remembering once off-screen
NOTABLE = ("tree", "water", "stone", "coal", "iron", "diamond", "table", "furnace", "lava",
           "cow", "zombie", "skeleton", "shade", "arrow", "sapling", "ripe_plant")
REMEMBER_ALL = ("table", "furnace", "coal", "iron", "diamond", "sapling", "ripe_plant")
REMEMBER_LAST = ("tree", "water", "stone", "cow")

MAP_CHARS = {
    "grass": "G", "sand": "S", "path": "P", "water": "W", "tree": "T", "stone": "R", "coal": "C",
    "iron": "I", "diamond": "D", "lava": "L", "table": "B", "furnace": "F", "player": "Y",
    "cow": "O", "zombie": "Z", "skeleton": "K", "arrow": "A", "sapling": "N", "ripe_plant": "E",
    "void": "X", "shade": "V", "ally": "M", "downed_ally": "H", "campfire": "Q", "campfire_low": "Q", "campfire_out": "q",
    "stone_campfire": "Q", "stone_campfire_low": "Q", "stone_campfire_out": "q",
    "placed_wood_bucket": "U", "placed_iron_bucket": "U", "placed_wood_bucket_empty": "u", "placed_iron_bucket_empty": "u",
}
DROPPED_CHAR = "J"  # any dropped item; which item it is appears under "nearest"


def map_char(cell):
    return MAP_CHARS.get(cell, DROPPED_CHAR)


def notable(cell):
    return cell in NOTABLE or cell.endswith("ally") or cell.startswith(("dropped_", "placed_")) or "campfire" in cell


def obj_name(obj):
    if hasattr(obj, "percept_name"):      # multiplayer objects (other players, dropped items) name themselves
        return obj.percept_name
    if isinstance(obj, objects.Plant):
        return "ripe_plant" if obj.ripe else "sapling"
    return type(obj).__name__.lower()


POSITION_WORDS = {"screen": ("right", "left", "down", "up"), "compass": ("east", "west", "south", "north")}
POSITION_STYLE = "screen"  # multiplayer switches to "compass" so positions use the same words as speech


def rel(dx, dy):
    east, west, south, north = POSITION_WORDS[POSITION_STYLE]
    parts = []
    if dx:
        parts.append("%d %s" % (abs(dx), east if dx > 0 else west))
    if dy:
        parts.append("%d %s" % (abs(dy), south if dy > 0 else north))
    return " ".join(parts) or "here"


def local_grid(env):
    """grid[x][y] cell names for the 9x7 map area the player sees; objects override materials."""
    world, p = env._world, env._player
    grid = [["void"] * VIEW_H for _ in range(VIEW_W)]
    for x in range(VIEW_W):
        for y in range(VIEW_H):
            px, py = int(p.pos[0]) + x - CX, int(p.pos[1]) + y - CY
            if 0 <= px < world.area[0] and 0 <= py < world.area[1]:
                mat, obj = world[(px, py)]
                grid[x][y] = obj_name(obj) if obj is not None else mat
    grid[CX][CY] = "player"  # the centre is always you, however other players are named
    return grid


class Observer:
    """Builds percepts and keeps the agent's memory for one episode."""

    def __init__(self, env, max_events=6, event_horizon=60):
        self.env = env
        self.max_events = max_events
        self.event_horizon = event_horizon
        self.events = []          # [step, text, count]
        self.known = {k: set() for k in REMEMBER_ALL}
        self.last_seen = {}
        self.step = 0
        self.stuck = 0
        self._prev = None
        self.percept = self._build(None)

    # -- event log ---------------------------------------------------------------------------
    def _log(self, text, at=None):
        at = self.step if at is None else at
        if self.events and self.events[-1][1] == text and at - self.events[-1][0] <= 3:
            self.events[-1][0] = at
            self.events[-1][2] += 1
        else:
            self.events.append([at, text, 1])
        self.events = self.events[-24:]

    def _diff(self, prev, cur, action):
        pi, ci = prev["inv"], cur["inv"]
        progressed = False
        for name in ACHIEVEMENTS:
            if cur["ach_counts"][name] > prev["ach_counts"][name]:
                self._log(name.replace("_", " "))
                progressed = True
        dh = ci["health"] - pi["health"]
        if dh < 0:
            flat = [c for col in cur["grid"] for c in col]
            next_to = lambda kind: any(kind in (prev["grid"][CX + dx][CY + dy], cur["grid"][CX + dx][CY + dy]) for dx, dy in DIRS.values())
            zombie_adjacent, shade_adjacent = next_to("zombie"), next_to("shade")
            drained = [n for k, n in (("food", "starving"), ("drink", "thirst"), ("energy", "exhaustion")) if ci[k] == 0]
            # a monster hit costs 2+ health, running on empty costs exactly 1
            if shade_adjacent and (dh <= -2 or not drained):
                cause = "shade attack"
            elif zombie_adjacent and (dh <= -2 or not drained):
                cause = "zombie attack"
            elif ("arrow" in flat or "skeleton" in flat) and (dh <= -2 or not drained):
                cause = "skeleton arrow"
            elif drained:
                cause = drained[0]
            else:
                cause = "unknown cause"
            self._log("lost %d health from %s" % (-dh, cause))
        if cur["sleeping"] and not prev["sleeping"]:
            self._log("fell asleep")
        if prev["sleeping"] and not cur["sleeping"]:
            self._log("woke up")
        if action is not None and not prev["sleeping"]:
            name = ACTIONS[action]
            target = prev["facing_cell"]
            if name == "do" and target in ("zombie", "skeleton", "shade", "cow") and not progressed:
                self._log("hit " + target)
                progressed = True
            elif name.startswith(("make_", "place_")) and not progressed:
                self._log(name.replace("_", " ") + " failed")
            elif name == "do" and target == "downed_ally":
                self._log("revived a teammate")             # nothing of the reviver's own changes, but it worked
            elif name == "do" and not progressed and ci == pi and target not in ("water", "grass"):
                self._log("do on %s had no effect" % target)
            elif name.startswith("move_") and cur["pos"] == prev["pos"] and target_blocked(prev, name):
                pass  # a turn, not worth logging
        moved = cur["pos"] != prev["pos"]
        self.stuck = 0 if (moved or progressed or ci != pi or cur["sleeping"]) else self.stuck + 1

    # -- percept -----------------------------------------------------------------------------
    def _build(self, action):
        env = self.env
        p = env._player
        grid = local_grid(env)
        pos = (int(p.pos[0]), int(p.pos[1]))
        facing = (int(p.facing[0]), int(p.facing[1]))
        cur = {
            "inv": dict(p.inventory),
            "ach_counts": dict(p.achievements),
            "ach": {k for k, v in p.achievements.items() if v > 0},
            "grid": grid,
            "pos": pos,
            "facing": facing,
            "facing_cell": grid[CX + facing[0]][CY + facing[1]],
            "sleeping": bool(p.sleeping),
            "daylight": float(env._world.daylight),
            "step": self.step,
        }
        # dawn and dusk look identical in one frame; the trend is something only memory can supply
        last_light = self._prev["daylight"] if self._prev else cur["daylight"] - 1e-6
        cur["light_trend"] = "rising" if cur["daylight"] >= last_light else "falling"
        # landmark memory: refresh from what is on screen
        for x in range(VIEW_W):
            for y in range(VIEW_H):
                g = (pos[0] + x - CX, pos[1] + y - CY)
                cell = grid[x][y]
                for k in REMEMBER_ALL:
                    if cell == k:
                        self.known[k].add(g)
                    else:
                        self.known[k].discard(g)
        # common terrain: remember only the closest instance from the last time it was on screen
        for k, (dx, dy) in nearest_in_view(grid).items():
            if k in REMEMBER_LAST:
                self.last_seen[k] = (pos[0] + dx, pos[1] + dy)
        for k, g in list(self.last_seen.items()):
            if self._in_view(g, pos) and grid[g[0] - pos[0] + CX][g[1] - pos[1] + CY] != k:
                del self.last_seen[k]
        remembered = {}
        for k in REMEMBER_ALL:
            out = [g for g in self.known[k] if not self._in_view(g, pos)]
            if out:
                g = min(out, key=lambda g: abs(g[0] - pos[0]) + abs(g[1] - pos[1]))
                remembered[k] = (g[0] - pos[0], g[1] - pos[1])
        for k, g in self.last_seen.items():
            if not self._in_view(g, pos):
                remembered[k] = (g[0] - pos[0], g[1] - pos[1])
        cur["remembered"] = remembered
        if self._prev is not None:
            self._diff(self._prev, cur, action)
        cur["stuck"] = self.stuck
        cur["events"] = [(self.step - s, t, n) for s, t, n in self.events
                         if self.step - s <= self.event_horizon][-self.max_events:]
        self._prev = cur
        return cur

    @staticmethod
    def _in_view(g, pos):
        return abs(g[0] - pos[0]) <= CX and abs(g[1] - pos[1]) <= CY

    def after_step(self, action):
        self.step += 1
        self.percept = self._build(action)
        return self.percept


def target_blocked(percept, move_name):
    dx, dy = DIRS[move_name[len("move_"):]]
    return percept["grid"][CX + dx][CY + dy] not in WALKABLE


def nearest_in_view(grid):
    best = {}
    for x in range(VIEW_W):
        for y in range(VIEW_H):
            cell = grid[x][y]
            if notable(cell):
                d = abs(x - CX) + abs(y - CY)
                if cell not in best or d < best[cell][0]:
                    best[cell] = (d, x - CX, y - CY)
    return {k: (v[1], v[2]) for k, v in sorted(best.items(), key=lambda kv: kv[1][0])}


def craft_range(grid):
    near = set()
    for x in range(CX - 1, CX + 2):
        for y in range(CY - 1, CY + 2):
            near.add(grid[x][y])
    return near


def state_dict(P, include_map=True, include_events=True, include_memory=True):
    inv = P["inv"]
    grid = P["grid"]
    near = craft_range(grid)
    light = P["daylight"]
    s = {k: inv[k] for k in STATS}
    s["inventory"] = {k: v for k, v in inv.items() if k not in STATS and v > 0}
    s["time"] = "day" if light > 0.84 else ("twilight" if light > 0.3 else "night")
    s["light"] = "%d%% %s" % (round(100 * light), P["light_trend"])
    if P["sleeping"]:
        s["asleep"] = True
    s["facing"] = "%s: %s" % (DIR_NAME[P["facing"]], P["facing_cell"])
    s["around"] = {d: grid[CX + dx][CY + dy] for d, (dx, dy) in DIRS.items()}
    s["table_nearby"] = "table" in near
    s["furnace_nearby"] = "furnace" in near
    s["nearest"] = {k: rel(*v) for k, v in nearest_in_view(grid).items()}
    if include_memory and P["remembered"]:
        s["remembered"] = {k: rel(*v) for k, v in P["remembered"].items()}
    if include_events:
        s["recent_events"] = ["%s ago: %s%s" % (ago, t, " x%d" % n if n > 1 else "") for ago, t, n in P["events"]]
    s["achievements_done"] = sorted(P["ach"])
    if include_map:
        s["map"] = [" ".join(map_char(grid[x][y]) for x in range(VIEW_W)) for y in range(VIEW_H)]
    return s


def state_text(P, **kw):
    return json.dumps(state_dict(P, **kw), separators=(",", ":"))


def crafter_score(success_rates_percent):
    s = np.asarray(success_rates_percent, dtype=np.float64)
    return float(np.exp(np.mean(np.log(1 + s))) - 1)
