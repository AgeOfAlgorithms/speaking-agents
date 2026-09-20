"""Rule-based Crafter player: the brain of the scripted teammate (mp.agents.ScriptedAgent).

It decides from the percept alone (see core.Observer) -- the same information a learned agent is
given as text -- plus a little private state (exploration heading, wander timer). No global map
access. It knows Crafter's original rules only; the cooperative behaviour is layered on top of it.
"""
from collections import deque

import numpy as np
from crafter import constants

from core import A, CX, CY, DIR_NAME, DIRS, TOOLS, VIEW_H, VIEW_W, WALKABLE, craft_range

OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}
TABLE_COST = constants.place["table"]["uses"]["wood"]
FURNACE_COST = constants.place["furnace"]["uses"]["stone"]
HOSTILE = ("zombie", "skeleton", "shade")
SOLID = {"stone", "coal", "iron", "diamond", "tree", "water", "table", "furnace"}  # monsters cannot pass
SEALABLE = set(constants.place["stone"]["where"])


def bfs(grid):
    """Shortest walkable paths from the player inside the visible area. Lava is never walked on."""
    dist, first = {(CX, CY): 0}, {(CX, CY): None}
    q = deque([(CX, CY)])
    while q:
        c = q.popleft()
        for name, (dx, dy) in DIRS.items():
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < VIEW_W and 0 <= n[1] < VIEW_H and n not in dist and grid[n[0]][n[1]] in WALKABLE:
                dist[n] = dist[c] + 1
                first[n] = first[c] or name
                q.append(n)
    return dist, first


class Teacher:
    def __init__(self, seed=0, shelter=False, topup=True, detour=True):
        self.use_shelter, self.use_topup, self.use_detour = shelter, topup, detour
        self.rng = np.random.RandomState(seed)
        self.heading = None
        self.wander = 0
        self.sapling_tries = 0
        self.plan = deque()  # committed action sequence (digging a night shelter)
        self.travel = 0   # consecutive steps spent walking toward a remembered place
        self.chase = 0    # consecutive steps spent chasing a monster

    def act(self, P):
        self._travelling = False
        name = self._decide(P)
        self.travel = self.travel + 1 if self._travelling else 0
        return A[name]

    # -- primitives --------------------------------------------------------------------------
    @staticmethod
    def cells(P, *names):
        g = P["grid"]
        return {(x, y) for x in range(VIEW_W) for y in range(VIEW_H) if g[x][y] in names}

    @staticmethod
    def mineable(P, cell):
        req = constants.collect.get(cell, {}).get("require")
        if cell not in ("stone", "coal", "iron", "diamond") or req is None:
            return False
        return all(P["inv"][k] >= v for k, v in req.items())

    def approach(self, P, targets):
        """Step toward the nearest target. 'READY' once it is directly faced, None if unreachable.
        Targets must be non-walkable cells (resources, creatures), so stepping at them only turns."""
        if not targets:
            return None
        f = P["facing"]
        if (CX + f[0], CY + f[1]) in targets:
            return "READY"
        for name, (dx, dy) in DIRS.items():
            if (CX + dx, CY + dy) in targets:
                return "move_" + name
        dist, first = bfs(P["grid"])
        best = None
        for tx, ty in targets:
            for dx, dy in DIRS.values():
                c = (tx + dx, ty + dy)
                if c in dist and (best is None or dist[c] < best[0]):
                    best = (dist[c], first[c])
        return "move_" + best[1] if best else None

    def use(self, P, targets):
        r = self.approach(P, targets)
        return "do" if r == "READY" else r

    def goto_offset(self, P, off):
        """Head for a remembered place outside the view; dig or bridge when the way is blocked."""
        self._travelling = True
        if self.travel > 60:  # not getting there (oscillating round an obstacle): give up for a while
            self.travel, self.wander = 0, 15
            return self.explore(P, dig=True, fresh=True)
        dist, first = bfs(P["grid"])
        best = min(dist, key=lambda c: (abs(off[0] - (c[0] - CX)) + abs(off[1] - (c[1] - CY)), dist[c]))
        if best != (CX, CY):
            return "move_" + first[best]
        horizontal = abs(off[0]) >= abs(off[1])
        d = ("right" if off[0] > 0 else "left") if horizontal else ("down" if off[1] > 0 else "up")
        r = self.push_through(P, d)
        if r:
            return r
        # Walled off. One random step would just bounce back here, so commit to a real detour,
        # starting sideways along the obstacle.
        if not self.use_detour:
            return self.explore(P, fresh=True)
        self.wander = 10
        g = P["grid"]
        side = [k for k in (("up", "down") if horizontal else ("left", "right"))
                if g[CX + DIRS[k][0]][CY + DIRS[k][1]] in WALKABLE]
        self.heading = side[self.rng.randint(len(side))] if side else None
        return self.explore(P)

    def push_through(self, P, d):
        dx, dy = DIRS[d]
        cell = P["grid"][CX + dx][CY + dy]
        facing_it = P["facing"] == (dx, dy)
        if self.mineable(P, cell):
            return "do" if facing_it else "move_" + d
        if cell == "water" and P["inv"]["stone"] > 0 and P["inv"]["wood_pickaxe"] > 0:
            return "place_stone" if facing_it else "move_" + d
        return None

    def explore(self, P, dig=False, fresh=False):
        g = P["grid"]
        free = [d for d, (dx, dy) in DIRS.items() if g[CX + dx][CY + dy] in WALKABLE]
        if fresh or self.heading is None:
            self.heading = None
        if self.heading in free and self.rng.rand() > 0.06:
            return "move_" + self.heading
        if dig and self.heading is not None and self.rng.rand() < 0.7:
            r = self.push_through(P, self.heading)
            if r:
                return r
        if not free:
            for d in DIRS:
                r = self.push_through(P, d)
                if r:
                    return r
            return "noop"
        choices = [d for d in free if self.heading is None or d != OPPOSITE[self.heading]] or free
        self.heading = choices[self.rng.randint(len(choices))]
        return "move_" + self.heading

    def place(self, P, what, keep_near=None):
        where = constants.place[what]["where"]
        g = P["grid"]
        if P["facing_cell"] in where:
            return "place_" + what
        options = []
        for d, (dx, dy) in DIRS.items():
            a, b = g[CX + dx][CY + dy], g[CX + 2 * dx][CY + 2 * dy]
            if a in WALKABLE and b in where:
                if keep_near:
                    around = {g[x][y] for x in range(CX + dx - 1, CX + dx + 2) for y in range(CY + dy - 1, CY + dy + 2)}
                    if keep_near not in around:
                        continue
                options.append(d)
        if options:
            return "move_" + options[self.rng.randint(len(options))]
        if self.mineable(P, P["facing_cell"]):
            return "do"
        return self.explore(P, fresh=True)

    # -- night shelter -----------------------------------------------------------------------
    # A zombie hits a sleeper for 7, and at night they come in groups, so nights are spent sealed in.
    # You can only place a block on the tile you face, and you only face an open tile by walking at
    # it, so a 1-cell hole cannot be closed from inside. A 2-cell tunnel with solid sides can: walk
    # to the far end, step back (now facing the entrance) and seal it.
    @staticmethod
    def enclosed(P):
        g = P["grid"]
        dist, _ = bfs(g)
        if len(dist) > 4:
            return False
        for x, y in dist:
            for dx, dy in DIRS.values():
                n = (x + dx, y + dy)
                if not (0 <= n[0] < VIEW_W and 0 <= n[1] < VIEW_H):
                    return False
                if n not in dist and g[n[0]][n[1]] not in SOLID:
                    return False
        return True

    def seek_shelter(self, P):
        g, inv = P["grid"], P["inv"]
        dist, first = bfs(g)
        best = None
        for (px, py), d in dist.items():
            for f, (dx, dy) in DIRS.items():
                line = [(px + k * dx, py + k * dy) for k in (1, 2, 3)]
                sides = [(c[0] + s * dy, c[1] + s * dx) for c in line[:2] for s in (1, -1)]
                if not all(0 <= x < VIEW_W and 0 <= y < VIEW_H for x, y in line + sides):
                    continue
                x1, x2, x3 = (g[x][y] for x, y in line)
                if not all(c in WALKABLE or self.mineable(P, c) for c in (x1, x2)):
                    continue
                if not all(g[x][y] in SOLID for x, y in sides):
                    continue
                if x3 not in SOLID and x3 not in SEALABLE:
                    continue
                stones_needed = 1 + (x3 not in SOLID)
                if inv["stone"] + (x1 == "stone") + (x2 == "stone") < stones_needed:
                    continue
                if best is None or d < best[0]:
                    best = (d, (px, py), f, x1, x2, x3)
        if best is None:
            if inv["wood_pickaxe"] and "stone" in P["remembered"]:
                return self.goto_offset(P, P["remembered"]["stone"])
            return None
        d, cell, f, x1, x2, x3 = best
        if d > 0:
            return "move_" + first[cell]
        plan = []
        for c in (x1, x2):
            if c not in WALKABLE:
                if not plan and P["facing"] != DIRS[f]:
                    plan.append("move_" + f)  # blocked, so this only turns us to face the rock
                plan.append("do")
            plan.append("move_" + f)
        if x3 not in SOLID:
            plan.append("place_stone")
        plan += ["move_" + OPPOSITE[f], "place_stone"]
        self.plan = deque(plan)
        return self.plan.popleft()

    # -- policy ------------------------------------------------------------------------------
    def _decide(self, P):
        inv, g, ach, mem = P["inv"], P["grid"], P["ach"], P["remembered"]
        if P["sleeping"]:
            return "sleep"
        fcell = P["facing_cell"]
        near = craft_range(g)

        # 1. anything hostile next to us: face it and fight
        if fcell in HOSTILE:
            self.plan.clear()
            return "do"
        for d, (dx, dy) in DIRS.items():
            if g[CX + dx][CY + dy] in HOSTILE:
                self.plan.clear()
                return "move_" + d
        hostile_in_view = bool(self.cells(P, *HOSTILE))

        # 1b. nightfall: get sealed in, then sleep until morning
        if self.plan:
            return self.plan.popleft()
        light, falling = P["daylight"], P["light_trend"] == "falling"
        enclosed = self.enclosed(P)
        critical = min(inv["food"], inv["drink"]) <= 2   # hunger and thirst outrank hiding
        night = (light < 0.75 or (falling and light < 0.85)) and self.use_shelter  # zombies spawn below ~0.84
        if night and enclosed and (inv["health"] > 3 or min(inv["food"], inv["drink"]) > 0):
            # stay put even when hungry: an empty stomach costs far less than the zombies outside
            return "sleep" if inv["energy"] < 9 else "noop"
        if night and not critical and not enclosed:
            r = self.seek_shelter(P)
            if r:
                return r
        if enclosed:  # morning (or starving): dig back out, preferring the way we are already facing
            for d in [DIR_NAME[P["facing"]]] + list(DIRS):
                r = self.push_through(P, d)
                if r:
                    return r

        if P["stuck"] > 12 and self.wander == 0:
            self.wander, self.heading = 8, None
        if self.wander > 0:
            self.wander -= 1
            return self.explore(P, dig=True)

        # 2. survival needs
        water, cows, ripe = self.cells(P, "water"), self.cells(P, "cow"), self.cells(P, "ripe_plant")
        if fcell == "water" and inv["drink"] < 9:
            return "do"
        if water and (inv["drink"] <= 7 or "collect_drink" not in ach):
            r = self.use(P, water)
            if r:
                return r
        if ripe and (inv["food"] <= 7 or "eat_plant" not in ach):
            r = self.use(P, ripe)
            if r:
                return r
        if cows and (inv["food"] <= 7 or "eat_cow" not in ach):
            r = self.use(P, cows)
            if r:
                return r
        # Zombies only spawn below ~0.84 daylight and hit a sleeper for 7, so nap by day, never at night.
        light, rising = P["daylight"], P["light_trend"] == "rising"
        if not hostile_in_view:
            if light > 0.84 and inv["energy"] <= 6 and (rising or (light > 0.97 and inv["energy"] >= 5)):
                return "sleep"
            if inv["energy"] == 0:
                return "sleep"
        # top up before dusk so the night can be spent sealed in
        low = 6 if (falling and light < 0.99 and self.use_topup) else 4
        if inv["drink"] <= low:
            if "water" in mem:
                return self.goto_offset(P, mem["water"])
            if inv["drink"] <= 3:
                return self.explore(P)
        if inv["food"] <= low:
            if "cow" in mem:
                return self.goto_offset(P, mem["cow"])
            if inv["food"] <= 3:
                return self.explore(P)

        # 3. craft whatever we have the materials for
        missing = [t for t in TOOLS if inv[t] == 0]
        order = ["wood_pickaxe", "wood_sword", "stone_pickaxe", "stone_sword", "iron_pickaxe", "iron_sword"]
        craftable = [t for t in order if t in missing and
                     all(inv[k] >= v for k, v in constants.make[t]["uses"].items())]
        if craftable:
            t = craftable[0]
            needs_furnace = "furnace" in constants.make[t]["nearby"]
            if "table" in near and (not needs_furnace or "furnace" in near):
                return "make_" + t
            if needs_furnace and "table" in near:
                furnaces = self.cells(P, "furnace")
                if furnaces:
                    r = self.approach(P, furnaces)
                    if r and r != "READY":
                        return r
                elif inv["stone"] >= FURNACE_COST:
                    return self.place(P, "furnace", keep_near="table")
                craftable = []  # need more stone first: fall through to gathering
            else:
                anchor = "furnace" if needs_furnace else "table"
                spots = self.cells(P, anchor) or self.cells(P, "table")
                if spots:
                    r = self.approach(P, spots)
                    if r and r != "READY":
                        return r
                known = mem.get(anchor) or mem.get("table")
                far = abs(known[0]) + abs(known[1]) if known else 99
                if far > 10 and not needs_furnace and inv["wood"] >= TABLE_COST + 1:
                    return self.place(P, "table")  # cheaper to build a new table here than walk back
                if far <= 25:
                    return self.goto_offset(P, known)
                if inv["wood"] >= TABLE_COST + 1:
                    return self.place(P, "table")

        # Afternoon: if no rock is on screen, start walking to where we last saw some, so that there
        # is something to dig into when the zombies come out.
        if (self.use_shelter and falling and light < 0.97 and not night and not critical and inv["wood_pickaxe"]
                and not self.cells(P, "stone") and "stone" in mem and min(inv["food"], inv["drink"]) > 4):
            return self.goto_offset(P, mem["stone"])
        # 4. gather
        wood_target = min(9, len(missing) + TABLE_COST)
        furnace_known = "furnace" in near or self.cells(P, "furnace") or "furnace" in mem
        stone_target = min(9, (inv["stone_pickaxe"] == 0) + (inv["stone_sword"] == 0) + 2
                           + (0 if furnace_known else FURNACE_COST) + ("place_stone" not in ach))
        ore_target = (inv["iron_pickaxe"] == 0) + (inv["iron_sword"] == 0)
        wants = []
        if inv["iron_pickaxe"]:
            wants.append(("diamond", True))
        if inv["stone_pickaxe"]:
            wants.append(("iron", inv["iron"] < ore_target))
        if inv["wood_pickaxe"]:
            wants.append(("coal", inv["coal"] < ore_target))
            wants.append(("stone", inv["stone"] < stone_target))
        wants.append(("tree", inv["wood"] < wood_target))
        for name, wanted in wants:
            if wanted:
                r = self.use(P, self.cells(P, name))
                if r:
                    return r

        # 5. cheap achievements along the way
        if "place_stone" not in ach and inv["stone"] > 1 and fcell in constants.place["stone"]["where"]:
            return "place_stone"
        if inv["sapling"] > 0 and fcell == "grass":
            return "place_plant"
        if "collect_sapling" not in ach and fcell == "grass" and self.sapling_tries < 15:
            self.sapling_tries += 1
            return "do"
        hunt = self.cells(P, *[h for h in ("zombie", "skeleton") if "defeat_" + h not in ach])   # only these count as achievements
        if hunt and inv["health"] >= 6 and self.chase < 30:
            r = self.use(P, hunt)
            if r:
                self.chase += 1
                return r
        self.chase = 0 if not hunt else self.chase

        # 6. travel toward the next thing we need, else explore
        if inv["wood_pickaxe"] == 0 or inv["wood"] < 2:
            goal = "tree"
        elif inv["stone_pickaxe"] == 0 or inv["stone_sword"] == 0 or inv["stone"] < stone_target:
            goal = "stone"
        elif inv["iron"] < ore_target and "iron" in mem:
            goal = "iron"
        elif inv["coal"] < ore_target and "coal" in mem:
            goal = "coal"
        elif inv["iron_pickaxe"] and "diamond" in mem:
            goal = "diamond"
        else:
            goal = None
        if goal in mem:
            return self.goto_offset(P, mem[goal])
        return self.explore(P, dig=inv["wood_pickaxe"] > 0 and goal != "tree")
