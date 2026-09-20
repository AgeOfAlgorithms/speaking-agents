"""Multiplayer Crafter: several players in one world, built to make cooperation worth something.

Same engine and rules as Crafter (world generation, creatures, crafting, day/night). What is new:

  players     N players spawn scattered around the start area. Monsters go for the nearest standing one.
  sharing     `drop_<item>` puts one item on the tile you face; `do` on a dropped item picks it up.
  buckets     `make_wood_bucket` (5 waters) / `make_iron_bucket` (12 waters). `do` on water drinks AND
              fills a carried bucket (two turns to fill). One water fully quenches one player:
              `drink_bucket` from the one you carry, or `place_bucket` it at your base so that anyone can
              `do` on it to drink. `do` on a placed bucket when you are not thirsty (or it is empty)
              picks it up.
  shades      a night-only monster that spawns in the dark on cave floor and sand (where zombies never do),
              hunts the nearest player and dissolves at dawn. It cannot enter firelight.
  campfire    a home base: while it burns no zombie or shade spawns within 6 tiles or steps within 3,
              so beside a fire you can sleep. Skeletons (halved in number) ignore fire: camp away from tunnels. `place_campfire` (3 wood) burns for a third of the night;
              `place_stone_campfire` (3 wood + 3 stone) burns three times as long, the whole night.
              `do` on either with wood feeds it: 3 wood refills it completely. A fire that is about to
              go out looks different (campfire_low), so someone has to stay awake or wake up to feed it.
  scaling     more players, more zombies: night spawn targets grow by half per extra standing player.
  revive      at 0 health you are DOWNED, not dead: you cannot act (you can still call for help) and
              die after `downed_steps` unless a teammate faces you and uses `do`.
  speech      a player may say one sentence (mp.language) INSTEAD of acting: speaking takes the turn,
              so talk is never free and chatter costs progress. Every other living player within
              `hear_radius` tiles (a circle) hears it next step, with the speaker's name and relative position.
              Hearing reaches further than the 9x7 view on purpose: if you could only hear players you
              can already see, speech would carry nothing you do not already know.
  reward      info["reward"][pid] is Crafter's individual reward. info["team_reward"] pays +1 to the
              whole team the FIRST time anyone unlocks an achievement (so dividing the work pays and
              three players doing the same thing does not), plus the team's mean health change.
"""
import math
import pathlib

import imageio.v3 as imageio
import numpy as np
from crafter import constants, engine, objects, worldgen

import core
from core import Observer, rel
from mp import language

BUCKETS = {"wood_bucket": 5, "iron_bucket": 12}  # capacity in waters
constants.items.setdefault("wood_bucket", {"max": 1, "initial": 0})
constants.items.setdefault("iron_bucket", {"max": 1, "initial": 0})
constants.make.setdefault("wood_bucket", {"uses": {"wood": 3}, "nearby": ["table"], "gives": 1})
constants.make.setdefault("iron_bucket", {"uses": {"iron": 1, "coal": 1}, "nearby": ["table", "furnace"], "gives": 1})

DROPPABLE = ["sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
             "wood_sword", "stone_sword", "iron_sword"]


def _daylight(step):
    return 1 - abs(math.cos(math.pi * ((step / 300) % 1 + 0.3))) ** 3


NIGHT_STEPS = sum(_daylight(s) < 0.75 for s in range(300))    # the dark part of one 300-step day (about 170 steps)
# A wood campfire burns for a third of the night, a stone campfire three times as long: the whole night.
# Either is refilled to full by the same 3 wood; each wood adds a third of that fire's burn time.
CAMPFIRE = {"no_spawn": 6, "no_entry": 3, "refuel_wood": 3, "low": 15, "night_steps": NIGHT_STEPS,
            "kinds": {"campfire": {"cost": {"wood": 3}, "burn": round(NIGHT_STEPS / 3)},
                      "stone_campfire": {"cost": {"wood": 3, "stone": 3}, "burn": NIGHT_STEPS}}}
ACTIONS = (list(constants.actions) + ["make_wood_bucket", "make_iron_bucket", "drink_bucket", "place_bucket", "place_campfire",
                                      "place_stone_campfire"]
           + ["drop_" + k for k in DROPPABLE])
A = {n: i for i, n in enumerate(ACTIONS)}
# Players get random names each game (drawn with the world's seeded RNG, so a seed reproduces them).
# The names appear in event logs, speech and the text state, and changing them every episode keeps a
# learning agent from treating "Bo" as a fixed symbol rather than as whoever its teammate is.
NAME_POOL = ["Ada", "Ben", "Cleo", "Dara", "Eli", "Faye", "Gus", "Hana", "Ivo", "Jun", "Kira", "Leo", "Mina", "Nico",
             "Opal", "Pia", "Quinn", "Rafa", "Suki", "Tomas", "Uma", "Vera", "Wren", "Xavi", "Yara", "Zeke", "Amir",
             "Bea", "Cato", "Devi", "Emil", "Freya", "Gael", "Hugo", "Ines", "Joss", "Kofi", "Lena", "Milo", "Nora"]
COMPASS = {"left": "west", "right": "east", "up": "north", "down": "south"}  # north is up on screen


class Item(objects.Object):
    """Something lying on the ground. Occupies its tile, like every Crafter object."""

    def __init__(self, world, pos, kind, amount=1):
        super().__init__(world, pos)
        self.kind, self.amount = kind, amount
        self.health = 1

    texture = property(lambda self: "dropped_" + self.kind)     # icon on a shadow, not the raw block texture
    percept_name = property(lambda self: "dropped_" + self.kind)

    def update(self):
        self.health = 1  # arrows and the like cannot destroy it


class Bucket(objects.Object):
    """A bucket set down as a shared water station."""

    def __init__(self, world, pos, kind, water, owner):
        super().__init__(world, pos)
        self.kind, self.water, self.owner = kind, water, owner
        self.health = 1

    texture = property(lambda self: self.kind + ("_full" if self.water > 0 else ""))
    percept_name = property(lambda self: "placed_" + self.kind + ("" if self.water > 0 else "_empty"))

    def update(self):
        self.health = 1


class Campfire(objects.Object):
    def __init__(self, world, pos, kind="campfire"):
        super().__init__(world, pos)
        self.kind, self.burn = kind, CAMPFIRE["kinds"][kind]["burn"]
        self.fuel, self.health = self.burn, 1

    lit = property(lambda self: self.fuel > 0)

    @property
    def percept_name(self):
        """campfire / campfire_low (about to go out: feed it) / campfire_out -- likewise stone_campfire."""
        return self.kind + ("" if self.fuel > CAMPFIRE["low"] else "_low" if self.fuel > 0 else "_out")

    texture = percept_name

    def feed(self):
        self.fuel = min(self.burn, self.fuel + math.ceil(self.burn / CAMPFIRE["refuel_wood"]))

    def update(self):
        self.health = 1
        self.fuel = max(0, self.fuel - 1)


class MPPlayer(objects.Player):
    def __init__(self, world, pos, pid, name):
        super().__init__(world, pos)
        self.pid, self.name, self.alive = pid, name, True
        self.downed = 0                                    # steps left before a downed player dies
        self.revived_by = None                             # name of the teammate who just revived this player
        self.bucket_water = {k: 0 for k in BUCKETS}
        for k in BUCKETS:
            self.achievements["make_" + k] = 0             # not part of Crafter's 22, tracked separately
        self.stats = {"dropped": 0, "picked_up": 0, "revives": 0, "times_downed": 0, "shared_drinks": 0, "campfires": 0, "shades_defeated": 0}
        self.fires = []                                    # the shared list of campfires, set by the env

    @property
    def percept_name(self):
        return "downed_ally" if self.downed else "ally"

    @property
    def texture(self):
        """Each player slot has its own shirt colour (mp/make_assets.py), plus a downed sprite."""
        pose = "player-downed" if self.downed else super().texture
        return "p%d-%s" % (self.pid % 6, pose)

    @property
    def standing(self):
        return self.alive and not self.downed

    def update(self):
        if self.downed:
            self.inventory["health"] = 0
            self.downed -= 1
            if self.downed == 0:
                self.alive = False
            return
        a = self.action
        special = a.startswith("drop_") or a in ("drink_bucket", "place_bucket", "place_campfire", "place_stone_campfire")
        if special and not self.sleeping:
            if a.startswith("drop_"):
                self._drop(a[len("drop_"):])
            else:
                getattr(self, "_" + a)()
        if special:
            self.action = "noop"
        super().update()

    def _facing_target(self):
        target = (self.pos[0] + self.facing[0], self.pos[1] + self.facing[1])
        return (target,) + self.world[target]

    # -- sharing ---------------------------------------------------------------------------------
    def _drop(self, kind):
        if self.inventory.get(kind, 0) <= 0:
            return
        target, material, obj = self._facing_target()
        if isinstance(obj, Item) and obj.kind == kind:
            obj.amount += 1
        elif obj is None and material in constants.walkable:
            self.world.add(Item(self.world, target, kind))
        else:
            return
        self.inventory[kind] -= 1
        self.stats["dropped"] += 1

    # -- buckets ---------------------------------------------------------------------------------
    def _do_material(self, target, material):
        super()._do_material(target, material)          # on water this is the normal Crafter sip
        if material == "water":
            self._fill_bucket()

    def _place_campfire(self, kind="campfire"):
        target, material, obj = self._facing_target()
        cost = CAMPFIRE["kinds"][kind]["cost"]
        if obj is None and material in constants.walkable and all(self.inventory[k] >= v for k, v in cost.items()):
            for k, v in cost.items():
                self.inventory[k] -= v
            fire = Campfire(self.world, target, kind)
            self.world.add(fire)
            self.fires.append(fire)
            self.stats["campfires"] += 1

    def _place_stone_campfire(self):
        self._place_campfire("stone_campfire")

    def _fill_bucket(self):
        for kind in ("iron_bucket", "wood_bucket"):
            cap = BUCKETS[kind]
            if self.inventory[kind] and self.bucket_water[kind] < cap:
                self.bucket_water[kind] = min(cap, self.bucket_water[kind] + math.ceil(cap / 2))  # two turns to fill
                return

    def _quench(self):
        self.inventory["drink"] = constants.items["drink"]["max"]
        self._thirst = 0
        self.achievements["collect_drink"] += 1

    def _drink_bucket(self):
        if self.inventory["drink"] >= constants.items["drink"]["max"]:
            return
        for kind in ("wood_bucket", "iron_bucket"):
            if self.inventory[kind] and self.bucket_water[kind] > 0:
                self.bucket_water[kind] -= 1
                self._quench()
                return

    def _place_bucket(self):
        target, material, obj = self._facing_target()
        if obj is not None or material not in constants.walkable:
            return
        held = [k for k in BUCKETS if self.inventory[k]]
        if not held:
            return
        kind = max(held, key=lambda k: self.bucket_water[k])
        self.world.add(Bucket(self.world, target, kind, self.bucket_water[kind], self.pid))
        self.inventory[kind] -= 1
        self.bucket_water[kind] = 0

    def _do_object(self, obj):
        if isinstance(obj, Item):
            take = min(obj.amount, constants.items[obj.kind]["max"] - self.inventory[obj.kind])
            if take > 0:
                self.inventory[obj.kind] += take
                self.stats["picked_up"] += take
                obj.amount -= take
                if obj.amount <= 0:
                    self.world.remove(obj)
        elif isinstance(obj, Bucket):
            if obj.water > 0 and self.inventory["drink"] < constants.items["drink"]["max"]:
                obj.water -= 1
                self._quench()
                self.stats["shared_drinks"] += obj.owner != self.pid
            elif not self.inventory[obj.kind]:
                self.inventory[obj.kind] = 1
                self.bucket_water[obj.kind] = obj.water
                self.world.remove(obj)
        elif isinstance(obj, Shade):
            obj.health -= max(1, self.inventory["wood_sword"] and 2, self.inventory["stone_sword"] and 3, self.inventory["iron_sword"] and 5)
            self.stats["shades_defeated"] += obj.health <= 0
        elif isinstance(obj, Campfire):
            if self.inventory["wood"] > 0 and obj.fuel < obj.burn:
                self.inventory["wood"] -= 1
                obj.feed()
        elif isinstance(obj, MPPlayer):
            if obj.downed:                                   # revive; no friendly fire otherwise
                obj.downed = 0
                obj.inventory["health"] = 3
                for k in ("food", "drink", "energy"):
                    obj.inventory[k] = max(obj.inventory[k], 3)
                obj._last_health = obj.health
                obj.revived_by = self.name
                self.stats["revives"] += 1
        else:
            super()._do_object(obj)


class _Nobody:
    pos, sleeping, health = np.array((-10 ** 6, -10 ** 6)), False, 0


class _Hunter:
    """Mixin: a monster's `player` is whichever standing player is nearest right now."""

    def _nearest(self):
        targets = [p for p in self._players if p.standing]
        return min(targets, key=self.distance) if targets else _Nobody

    player = property(lambda self: self._nearest(), lambda self, v: None)


class MPZombie(_Hunter, objects.Zombie):
    percept_name = "zombie"

    def __init__(self, world, pos, players, fires):
        self._players, self._fires = players, fires
        objects.Zombie.__init__(self, world, pos, None)

    def move(self, direction):
        target = self.pos + np.array(direction)
        if any(f.lit and np.abs(f.pos - target).max() <= CAMPFIRE["no_entry"] for f in self._fires):
            return False                                   # will not step into the firelight
        return super().move(direction)


class MPSkeleton(_Hunter, objects.Skeleton):
    """Cave archer, day or night. Fire means nothing to it -- do not camp beside a tunnel."""
    percept_name = "skeleton"

    def __init__(self, world, pos, players, fires=None):
        self._players = players
        objects.Skeleton.__init__(self, world, pos, None)


class Shade(_Hunter, objects.Object):
    """The night itself. Shades appear only in the dark, on cave floor and sand -- the ground zombies never
    spawn on -- so at night nowhere is safe except the firelight, which they cannot enter. They
    hunt the nearest standing player, hit like a zombie (2, or 7 to a sleeper), and dissolve at daybreak."""
    percept_name = texture = "shade"

    def __init__(self, world, pos, players, fires):
        super().__init__(world, pos)
        self._players, self._fires = players, fires
        self.health, self.cooldown = 3, 0

    def move(self, direction):
        target = self.pos + np.array(direction)
        if any(f.lit and np.abs(f.pos - target).max() <= CAMPFIRE["no_entry"] for f in self._fires):
            return False
        return super().move(direction)

    def update(self):
        if self.health <= 0 or self.world.daylight > 0.85:
            self.world.remove(self)
            return
        if self.distance(self.player) <= 10 and self.random.uniform() < 0.9:
            self.move(self.toward(self.player, self.random.uniform() < 0.8))
        else:
            self.move(self.random_dir())
        if self.distance(self.player) <= 1:
            if self.cooldown:
                self.cooldown -= 1
            else:
                self.player.health -= 7 if self.player.sleeping else 2
                self.cooldown = 5


class _Textures(engine.Textures):
    def __init__(self):
        super().__init__(constants.root / "assets")
        for filename in (pathlib.Path(__file__).parent / "assets").glob("*.png"):
            image = imageio.imread(filename.read_bytes())
            image = image.transpose((1, 0) + tuple(range(2, len(image.shape))))
            self._originals[filename.stem] = image
            self._textures[(filename.stem, image.shape[:2])] = image


class _View:
    """What core.Observer expects of an env: a world and 'the' player."""

    def __init__(self, world, player):
        self._world, self._player = world, player


class MPObserver(Observer):
    """Per-player percepts: core's percept plus teammates, buckets and what was heard."""

    def __init__(self, world, player, players, **kw):
        self.heard_now, self.comms, self._players = [], [], players
        self._down_logged = False
        super().__init__(_View(world, player), **kw)

    def _build(self, action):
        me = self.env._player
        if me.downed and not self._down_logged:
            self._log("you went down")
        self._down_logged = bool(me.downed)
        if me.revived_by:
            self._log("%s revived you" % me.revived_by)
            me.revived_by = None
        P = super()._build(action)
        P.update(name=me.name, heard=list(self.heard_now), downed=me.downed,
                 teammates=[q.name for q in self._players if q is not me and q.alive],
                 bucket_water={k: me.bucket_water[k] for k in BUCKETS if me.inventory[k]})
        P["team"] = [{"name": q.name, "offset": (int(q.pos[0] - me.pos[0]), int(q.pos[1] - me.pos[1])),
                      "downed": bool(q.downed), "asleep": bool(q.sleeping)} for q in self._players
                     if q is not me and q.alive and abs(q.pos[0] - me.pos[0]) <= core.CX and abs(q.pos[1] - me.pos[1]) <= core.CY]
        return P

    def hear(self, speaker, text, offset):
        now = self.step + 1        # speech is delivered before this observer's clock ticks for the step
        entry = {"step": now, "from": speaker, "text": text, "where": rel(*offset)}
        self.heard_now.append(entry)
        self.comms.append(entry)
        self._log('%s said "%s" (%s)' % (speaker, text, rel(*offset)), at=now)

    def said(self, text):
        self.comms.append({"step": self.step + 1, "from": "me", "text": text, "where": "here"})


def state_dict(P):
    """Text state for a multiplayer percept: core's single-player state plus the social parts."""
    s = {"you": P["name"], "teammates": P["teammates"], **core.state_dict(P)}
    g = P["grid"]
    s["facing"] = "%s: %s" % (COMPASS[core.DIR_NAME[P["facing"]]], P["facing_cell"])
    s["around"] = {COMPASS[d]: g[core.CX + dx][core.CY + dy] for d, (dx, dy) in core.DIRS.items()}
    if P["downed"]:
        s["downed"] = "you are down for %d more steps unless a teammate revives you" % P["downed"]
    if P["bucket_water"]:
        s["bucket_water"] = P["bucket_water"]
    if P["team"]:
        s["teammates_in_view"] = {t["name"]: rel(*t["offset"]) + (" (downed, needs revive)" if t["downed"] else " (asleep)" if t["asleep"] else "")
                                  for t in P["team"]}
    if P["heard"]:
        s["heard_now"] = ['%s: "%s" (%s)' % (h["from"], h["text"], h["where"]) for h in P["heard"]]
    return s


class MPEnv:
    def __init__(self, n_players=3, names=None, area=(64, 64), view=(9, 9), length=10000, seed=0,
                 hear_radius=16, spawn_spread=(4, 10), downed_steps=40):
        core.POSITION_STYLE = "compass"
        self.n = n_players
        self._fixed_names = list(names) if names else [None] * n_players   # None = pick a random name
        self._area, self._view, self._length, self._seed = area, np.array(view), length, seed
        self.hear_radius, self.spawn_spread = hear_radius, spawn_spread
        self.downed_steps = downed_steps
        self._world = engine.World(area, constants.materials, (12, 12))
        self._textures = _Textures()
        item_rows = int(np.ceil(len(constants.items) / view[0]))
        self._local_view = engine.LocalView(self._world, self._textures, [view[0], view[1] - item_rows])
        self._item_view = engine.ItemView(self._textures, [view[0], item_rows])
        self._episode = 0

    # -- setup ---------------------------------------------------------------------------------
    def reset(self):
        self._episode += 1
        self._step = 0
        world = self._world
        world.reset(seed=hash((self._seed, self._episode)) % (2 ** 31 - 1))
        self._update_time()
        center = (world.area[0] // 2, world.area[1] // 2)
        anchor = objects.Player(world, center)  # only its position is used, to shape the start area
        simplex = worldgen.opensimplex.OpenSimplex(seed=world.random.randint(0, 2 ** 31 - 1))
        tunnels = np.zeros(world.area, bool)
        for x in range(world.area[0]):
            for y in range(world.area[1]):
                worldgen._set_material(world, (x, y), anchor, tunnels, simplex)
        lo, hi = self.spawn_spread
        spots = [(x, y) for x in range(world.area[0]) for y in range(world.area[1])
                 if world[x, y][0] == "grass" and lo <= abs(x - center[0]) + abs(y - center[1]) <= hi]
        order = world.random.permutation(len(spots))
        chosen = []
        for i in order:  # spread the team out: nobody spawns within 5 tiles of a teammate
            if all(abs(spots[i][0] - c[0]) + abs(spots[i][1] - c[1]) >= 5 for c in chosen):
                chosen.append(spots[i])
            if len(chosen) == self.n:
                break
        while len(chosen) < self.n:
            chosen.append(spots[order[len(chosen)]])
        taken = {n for n in self._fixed_names if n}
        pool = [n for n in NAME_POOL if n not in taken]
        picks = iter(pool[i] for i in world.random.permutation(len(pool)))
        self.names = [n or next(picks) for n in self._fixed_names]
        self.players, self.fires = [], []
        for pid, pos in enumerate(chosen):
            p = MPPlayer(world, pos, pid, self.names[pid])
            p.fires = self.fires
            world.add(p)
            self.players.append(p)
        for x in range(world.area[0]):
            for y in range(world.area[1]):
                self._set_object((x, y), tunnels)
        self.observers = [MPObserver(world, p, self.players) for p in self.players]
        self._last_health = [p.health for p in self.players]
        self._unlocked = [set() for _ in self.players]
        self.team_unlocked = set()
        self.last_speech = {}
        return self._annotate([ob.percept for ob in self.observers])

    def _annotate(self, percepts):
        for p, P in zip(self.players, percepts):
            if p.alive:
                P["blocked"] = {a: why for a, why in self.available(p.pid).items() if why}
                P["use"] = self.use_hint(p.pid)
        return percepts

    def _near(self, pos):
        present = [p for p in self.players if p.alive]
        return min(np.abs(np.array(pos) - p.pos).sum() for p in present) if present else 1e9

    def _set_object(self, pos, tunnels):
        world, uniform = self._world, self._world.random.uniform
        material, obj = world[pos]
        if obj is not None or material not in constants.walkable:
            return
        dist = self._near(pos)
        if dist > 3 and material == "grass" and uniform() > 0.985:
            world.add(objects.Cow(world, pos))
        elif dist > 10 and uniform() > 0.993:
            world.add(MPZombie(world, pos, self.players, self.fires))
        elif material == "path" and tunnels[pos] and uniform() > 0.975:     # half of Crafter's skeleton density
            world.add(MPSkeleton(world, pos, self.players, self.fires))

    # -- step ----------------------------------------------------------------------------------
    def step(self, actions, speech=None):
        """actions: {pid: action index or name}; speech: {pid: slot indices (mp.language) or free text}."""
        self._step += 1
        self._update_time()
        speech = speech or {}
        present = [p for p in self.players if p.alive]
        texts = {}
        for p in present:
            said = speech.get(p.pid)
            if p.sleeping or said is None:
                continue
            # a human may say anything; agents use the vocabulary in mp.language
            text = " ".join(said.split())[:120] if isinstance(said, str) else ("" if language.is_silent(said) else language.render(said))
            if text:
                texts[p.pid] = text
        for p in present:
            a = actions.get(p.pid, "noop")
            p.action = ACTIONS[a] if not isinstance(a, str) else a
            if p.pid in texts:
                p.action = "noop"                          # speaking takes the turn
        for obj in self._world.objects:
            if self._near(obj.pos) < 2 * max(self._view):
                obj.update()
        if self._step % 10 == 0:
            for chunk, objs in self._world.chunks.items():
                self._balance_chunk(chunk, objs)

        # speech is delivered before observations are rebuilt, so it is heard on the very next step
        self.last_speech = {}
        for ob in self.observers:
            ob.heard_now = []
        for p in present:
            text = texts.get(p.pid)
            if not text or not p.alive:
                continue
            self.last_speech[p.pid] = text
            self.observers[p.pid].said(text)
            for q in self.players:
                if q is not p and q.alive and self.can_hear(q.pos, p.pos):
                    self.observers[q.pid].hear(p.name, text, tuple(int(v) for v in (p.pos - q.pos)))

        rewards, health_change, new_team, events = {}, [], set(), []
        for p in present:
            if p.alive and not p.downed and p.health <= 0:       # knocked down this step
                p.downed, p.sleeping = self.downed_steps, False
                p.stats["times_downed"] += 1
                events.append("%s is down" % p.name)
            dh = (p.health - self._last_health[p.pid]) / 10
            self._last_health[p.pid] = p.health
            got = {k for k, c in p.achievements.items() if c > 0 and k in constants.achievements} - self._unlocked[p.pid]
            self._unlocked[p.pid] |= got
            new_team |= got - self.team_unlocked
            rewards[p.pid] = dh + len(got)
            health_change.append(dh)
            if not p.alive:                                       # bled out
                self._world.remove(p)
                events.append("%s died" % p.name)
        self.team_unlocked |= new_team
        team_reward = len(new_team) + (float(np.mean(health_change)) if health_change else 0.0)

        percepts = []
        for p, ob in zip(self.players, self.observers):
            if p in present:
                a = "noop" if p.pid in texts else actions.get(p.pid, "noop")
                idx = A[a] if isinstance(a, str) else int(a)
                percepts.append(ob.after_step(idx if idx < len(constants.actions) else None))
            else:
                percepts.append(ob.percept)
        done = not any(p.standing for p in self.players) or self._step >= self._length
        info = {"reward": rewards, "team_reward": team_reward, "new_team_achievements": sorted(new_team),
                "alive": [p.alive for p in self.players], "speech": dict(self.last_speech), "events": events}
        return self._annotate(percepts), info, done

    def can_hear(self, listener_pos, speaker_pos):
        """Sound carries in a circle (the +0.5 rounds the tile grid off nicely), not a square."""
        d = np.asarray(listener_pos, dtype=float) - np.asarray(speaker_pos, dtype=float)
        return float(d @ d) <= (self.hear_radius + 0.5) ** 2

    def _update_time(self):
        progress = (self._step / 300) % 1 + 0.3
        self._world.daylight = 1 - np.abs(np.cos(np.pi * progress)) ** 3

    def _balance_chunk(self, chunk, objs):
        light = self._world.daylight
        scale = 1 + 0.5 * (sum(p.standing for p in self.players) - 1)       # more players, more zombies...
        dark = 3.5 - 3 * light
        zombies = dark * scale if dark >= 1 else dark                        # ...but only when it is dark: days stay safe
        self._balance_object(chunk, objs, objects.Zombie, "grass", 6, 0, min(0.9, 0.3 * scale), 0.4,
                             lambda pos: MPZombie(self._world, pos, self.players, self.fires),
                             lambda num, space: (0 if space < 50 else zombies, zombies))
        self._balance_object(chunk, objs, objects.Skeleton, "path", 7, 7, 0.05, 0.1,
                             lambda pos: MPSkeleton(self._world, pos, self.players),
                             lambda num, space: (0 if space < 6 else 1, 1))
        shades = 0.5 * zombies if dark >= 1 else 0                        # only in the dark
        for ground in ("path", "sand"):                                   # grass already has the zombies
            self._balance_object(chunk, objs, Shade, ground, 6, 0, min(0.9, 0.2 * scale), 0.5,
                                 lambda pos: Shade(self._world, pos, self.players, self.fires),
                                 lambda num, space: (0 if space < 6 else shades, shades))
        self._balance_object(chunk, objs, objects.Cow, "grass", 5, 5, 0.01, 0.1,
                             lambda pos: objects.Cow(self._world, pos),
                             lambda num, space: (0 if space < 30 else 1, 1.5 + light))

    def _balance_object(self, chunk, objs, cls, material, span_dist, despan_dist, spawn_prob, despawn_prob, ctor, target_fn):
        xmin, xmax, ymin, ymax = chunk
        random = self._world.random
        creatures = [obj for obj in objs if isinstance(obj, cls)]
        mask = self._world.mask(*chunk, material)
        target_min, target_max = target_fn(len(creatures), mask.sum())
        if len(creatures) < int(target_min) and random.uniform() < spawn_prob:
            xs = np.tile(np.arange(xmin, xmax)[:, None], [1, ymax - ymin])[mask]
            ys = np.tile(np.arange(ymin, ymax)[None, :], [xmax - xmin, 1])[mask]
            i = random.randint(0, len(xs))
            pos = np.array((xs[i], ys[i]))
            by_fire = cls in (objects.Zombie, Shade) and any(
                f.lit and np.abs(f.pos - pos).max() <= CAMPFIRE["no_spawn"] for f in self.fires)
            if self._world[pos][1] is None and self._near(pos) >= span_dist and not by_fire:
                self._world.add(ctor(pos))
        elif len(creatures) > int(target_max) and random.uniform() < despawn_prob:
            obj = creatures[random.randint(0, len(creatures))]
            if self._near(obj.pos) >= despan_dist:
                self._world.remove(obj)

    # -- what can this player do right now ---------------------------------------------------------
    def available(self, pid):
        """{action: None if it would work right now, else a short reason why not}. One source of truth
        for the GUI's greyed-out menus; the same table is an action mask for a learning agent."""
        p = self.players[pid]
        inv = p.inventory
        if not p.standing:
            return {a: "you are down" for a in ACTIONS}
        if p.sleeping:
            return {a: "you are asleep" for a in ACTIONS}
        _, material, obj = p._facing_target()
        nearby, _ = self._world.nearby(p.pos, 1)

        def lacking(uses):
            short = ["%d more %s" % (v - inv[k], k) for k, v in uses.items() if inv[k] < v]
            return "need " + ", ".join(short) if short else None

        def facing(where):
            return None if obj is None and material in where else "face an empty %s tile" % " / ".join(where)

        out = {a: None for a in ACTIONS}
        for name, info in constants.make.items():
            missing = [n for n in info["nearby"] if n not in nearby]
            out["make_" + name] = lacking(info["uses"]) or ("needs a %s nearby" % " and ".join(missing) if missing else None)
        for name, info in constants.place.items():
            out["place_" + name] = lacking(info["uses"]) or facing(info["where"])
        for kind, spec in CAMPFIRE["kinds"].items():
            out["place_" + kind] = lacking(spec["cost"]) or facing(constants.walkable)
        out["place_bucket"] = "you have no bucket" if not any(inv[k] for k in BUCKETS) else facing(constants.walkable)
        if not any(p.bucket_water[k] for k in BUCKETS if inv[k]):
            out["drink_bucket"] = "no water in a bucket you carry"
        elif inv["drink"] >= constants.items["drink"]["max"]:
            out["drink_bucket"] = "you are not thirsty"
        for kind in DROPPABLE:
            same_pile = isinstance(obj, Item) and obj.kind == kind
            out["drop_" + kind] = "you have none" if inv[kind] <= 0 else (None if same_pile else facing(constants.walkable))
        if inv["energy"] >= constants.items["energy"]["max"]:
            out["sleep"] = "you are not tired"
        return out

    def use_hint(self, pid):
        """What `do` would do for this player right now: (label, would_it_do_anything)."""
        p = self.players[pid]
        if not p.standing or p.sleeping:
            return "nothing", False
        inv = p.inventory
        _, material, obj = p._facing_target()
        if isinstance(obj, Item):
            return "pick up %s" % obj.kind.replace("_", " "), inv[obj.kind] < constants.items[obj.kind]["max"]
        if isinstance(obj, Bucket):
            if obj.water > 0 and inv["drink"] < constants.items["drink"]["max"]:
                return "drink from the bucket (%d left)" % obj.water, True
            return "pick up the bucket", not inv[obj.kind]
        if isinstance(obj, Campfire):
            state = "out" if not obj.lit else "dying down" if obj.fuel <= CAMPFIRE["low"] else "burning"
            return "add wood to the fire (%s)" % state, inv["wood"] > 0 and obj.fuel < obj.burn
        if isinstance(obj, MPPlayer):
            return ("revive %s" % obj.name, True) if obj.downed else (obj.name, False)
        if isinstance(obj, (objects.Zombie, objects.Skeleton, Shade)):
            return "attack the %s" % core.obj_name(obj), True
        if isinstance(obj, objects.Cow):
            return "hit the cow (food)", True
        if isinstance(obj, objects.Plant):
            return ("eat the plant", True) if obj.ripe else ("sapling, not ripe yet", False)
        if material == "water":
            fills = any(inv[k] and p.bucket_water[k] < cap for k, cap in BUCKETS.items())
            return "drink" + (" + fill bucket" if fills else ""), True
        if material == "tree":
            return "chop the tree", True
        if material == "grass":
            return "search the grass for a sapling", True
        rule = constants.collect.get(material)
        if rule:
            need = [k for k, v in rule["require"].items() if inv[k] < v]
            return ("mine %s" % material, True) if not need else ("mine %s: needs a %s" % (material, need[0].replace("_", " ")), False)
        return "nothing to use here", False

    # -- views ---------------------------------------------------------------------------------
    def render(self, pid, size=(144, 144)):
        size = np.array(size)
        unit = size // self._view
        p = self.players[pid]
        canvas = np.zeros(tuple(size) + (3,), np.uint8)
        view = np.concatenate([self._local_view(p, unit), self._item_bar(p, unit)], 1)
        border = (size - unit * self._view) // 2
        (x, y), (w, h) = border, view.shape[:2]
        canvas[x: x + w, y: y + h] = view
        return canvas.transpose((1, 0, 2))

    def _item_bar(self, p, unit):
        """Crafter's inventory bar (every item keeps its fixed slot; 18 slots, all 18 now in use), except
        that a bucket shows full/empty and the number beside it is the waters left, not 'x1'."""
        grid = self._item_view._grid
        canvas = np.zeros(tuple(grid * unit) + (3,), np.uint8)
        for index, (item, amount) in enumerate(p.inventory.items()):
            if amount < 1:
                continue
            texture, number = item, amount
            if item in BUCKETS:
                water = p.bucket_water[item]
                texture, number = item + ("_full" if water else ""), water or None
            self._item_view._item(canvas, index, texture, unit)
            if number is not None:
                pos = ((index % grid[0], index // grid[0]) * unit + 0.4 * unit).astype(np.int32)
                engine._draw_alpha(canvas, pos, self._textures.get(str(number), 0.6 * unit))
        return canvas

    def snapshot(self):
        """Everything a spectator GUI needs: the material map and where everyone and everything is."""
        w = self._world
        ents = []
        for obj in w.objects:
            if isinstance(obj, MPPlayer):
                continue
            e = {"kind": core.obj_name(obj), "x": int(obj.pos[0]), "y": int(obj.pos[1])}
            if isinstance(obj, Bucket):
                e["water"] = obj.water
            if isinstance(obj, Item):
                e["amount"] = obj.amount
            if isinstance(obj, Campfire):
                e["fuel"] = obj.fuel
            ents.append(e)
        return {
            "step": self._step, "daylight": float(w.daylight), "length": self._length,
            "materials": [w._mat_names[i] for i in range(len(w._mat_names))],
            "map": w._mat_map.T.tolist(), "entities": ents,
            "players": [{"pid": p.pid, "name": p.name, "alive": p.alive, "downed": int(p.downed),
                         "x": int(p.pos[0]), "y": int(p.pos[1]), "facing": [int(p.facing[0]), int(p.facing[1])],
                         "sleeping": bool(p.sleeping), "inventory": {k: int(v) for k, v in p.inventory.items()},
                         "bucket_water": {k: v for k, v in p.bucket_water.items() if p.inventory[k]},
                         "achievements": sorted(self._unlocked[p.pid]),
                         "stats": dict(p.stats), "saying": self.last_speech.get(p.pid)} for p in self.players],
            "team_achievements": sorted(self.team_unlocked),
        }
