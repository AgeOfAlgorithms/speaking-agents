"""Smoke tests for the multiplayer rules. Run: python -m mp.test_engine"""
import time

import numpy as np

from mp import language as L
from mp.engine import ACTIONS, CAMPFIRE, MPEnv, MPZombie, state_dict

WALK = ("grass", "sand", "path")


def beside(W, material):
    """A free grass tile next to `material`, and the direction that faces it."""
    for x in range(2, 62):
        for y in range(2, 62):
            if W[x, y][0] == material:
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (x - d[0], y - d[1])
                    if W[n][0] == "grass" and W[n][1] is None:
                        return n, d


def free_neighbour(W, pos):
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (int(pos[0]) + d[0], int(pos[1]) + d[1])
        if W[n][1] is None and W[n][0] in WALK:
            return n, (-d[0], -d[1])


def main():
    env = MPEnv(n_players=3, seed=7, names=["Ada", "Bo", "Cy"])
    env.reset()
    ada, bo, cy = env.players
    a, b = MPEnv(n_players=3, seed=11), MPEnv(n_players=3, seed=11)
    a.reset(), b.reset()
    c = MPEnv(n_players=3, seed=12, names=["Sean", None, None])
    c.reset()
    print("random names: seed 11 ->", a.names, "| same seed again ->", b.names, "| seed 12 with a chosen name ->", c.names)
    assert a.names == b.names and len(set(a.names)) == 3 and c.names[0] == "Sean"
    W = env._world
    print("actions:", len(ACTIONS))

    # do on water = sip + fill a carried bucket, two turns to fill
    spot, face = beside(W, "water")
    W.move(ada, spot)
    ada.facing = face
    ada.inventory["iron_bucket"], ada.inventory["drink"] = 1, 5
    env.step({0: "do"})
    mid = (ada.inventory["drink"], ada.bucket_water["iron_bucket"])
    env.step({0: "do"})
    print("do on water: drink 5 -> %d -> %d | iron bucket 0 -> %d -> %d of 12" % (
        mid[0], ada.inventory["drink"], mid[1], ada.bucket_water["iron_bucket"]))
    assert ada.bucket_water["iron_bucket"] == 12

    # one bucket water fully quenches
    ada.inventory["drink"] = 1
    env.step({0: "drink_bucket"})
    print("drink_bucket: drink 1 -> %d | bucket %d" % (ada.inventory["drink"], ada.bucket_water["iron_bucket"]))
    assert ada.inventory["drink"] == 9

    # campfire: place, refuel, zombies keep out
    spot2, face2 = free_neighbour(W, ada.pos)
    ada.facing = (-face2[0], -face2[1])
    ada.inventory["wood"] = 5
    env.step({0: "place_campfire"})
    fire = env.fires[0]
    print("campfire: wood 5 -> %d | fuel %d | Ada faces %s" % (ada.inventory["wood"], fire.fuel, env.observers[0].percept["facing_cell"]))
    wood_burn = CAMPFIRE["kinds"]["campfire"]["burn"]
    assert fire.fuel == wood_burn - 0 or fire.fuel == wood_burn - 1
    for _ in range(25):
        env.step({})
    low = env.observers[0].percept["facing_cell"]
    before = fire.fuel
    env.step({0: "do"})
    print("night is %d steps: wood fire burns %d, stone fire %d | after 25 steps fuel %d (%s) | +1 wood -> %d" % (
        CAMPFIRE["night_steps"], wood_burn, CAMPFIRE["kinds"]["stone_campfire"]["burn"], before, low, fire.fuel))
    assert CAMPFIRE["kinds"]["stone_campfire"]["burn"] == CAMPFIRE["night_steps"] and wood_burn == round(CAMPFIRE["night_steps"] / 3)
    ada.inventory.update(wood=3, stone=3)
    n2, f2 = free_neighbour(W, ada.pos)
    ada.facing = (-f2[0], -f2[1])
    env.step({0: "place_stone_campfire"})
    stone_fire = env.fires[-1]
    print("stone campfire: kind %s | fuel %d | wood %d stone %d left" % (stone_fire.kind, stone_fire.fuel, ada.inventory["wood"], ada.inventory["stone"]))
    assert stone_fire.kind == "stone_campfire" and ada.inventory["stone"] == 0
    for dx in (4, -4):
        zpos = (int(fire.pos[0]) + dx, int(fire.pos[1]))
        if W[zpos][1] is None and W[zpos][0] in WALK:
            z = MPZombie(W, zpos, env.players, env.fires)
            W.add(z)
            step_in = z.move((-1 if dx > 0 else 1, 0))
            print("zombie 4 tiles from the fire tries to step closer ->", step_in)
            assert step_in is False
            W.remove(z)
            break

    # downed -> call for help -> heard with a compass position -> revived
    cy.inventory["health"] = 0
    _, info, _ = env.step({})
    print("events:", info["events"], "| Cy downed for", cy.downed)
    W.move(ada, free_neighbour(W, bo.pos)[0])                       # bring Ada back within earshot
    _, info, _ = env.step({}, {2: L.encode(verb="help", object="me")})
    print("Cy calls:", info["speech"], "-> heard:", {ob.env._player.name: [h["text"] + " @ " + h["where"] for h in ob.percept["heard"]]
                                                       for ob in env.observers})
    n, facing = free_neighbour(W, cy.pos)
    W.move(ada, n)
    ada.facing = facing
    env.step({0: "noop"})
    print("Ada state:", {k: v for k, v in state_dict(env.observers[0].percept).items() if k in ("facing", "teammates_in_view")})
    env.step({0: "do"})
    print("revived: Cy downed=%d health=%d | Ada revives=%d" % (cy.downed, cy.health, ada.stats["revives"]))
    assert cy.downed == 0 and cy.health == 3

    # speaking takes the turn
    before = tuple(int(v) for v in bo.pos)
    free = next(d for d, (dx, dy) in {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}.items()
                if W[(before[0] + dx, before[1] + dy)][1] is None and W[(before[0] + dx, before[1] + dy)][0] in WALK)
    env.step({1: "move_" + free}, {1: "wait for me"})
    stayed = tuple(int(v) for v in bo.pos) == before
    env.step({1: "move_" + free})
    print("speaking takes the turn: stayed put while talking =", stayed, "| moved next turn =", tuple(int(v) for v in bo.pos) != before)
    assert stayed and tuple(int(v) for v in bo.pos) != before

    rng = np.random.RandomState(3)
    n, done, t0 = 0, False, time.time()
    while not done and n < 3000:
        sp = {i: tuple(int(rng.randint(k)) for k in L.SIZES) for i in range(3) if rng.rand() < 0.1}
        _, info, done = env.step({i: int(rng.randint(len(ACTIONS))) for i in range(3)}, sp)
        n += 1
    print("random game: %d steps at %.0f steps/s, no crash" % (n, n / (time.time() - t0)))
    print("ALL OK")


if __name__ == "__main__":
    main()
