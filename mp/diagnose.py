"""Where and when do scripted players go down? python -m mp.diagnose [scripted|scripted_silent] [games]"""
import sys
from collections import Counter

import numpy as np

from mp import agents
from mp.engine import MPEnv

team = sys.argv[1] if len(sys.argv) > 1 else "scripted"
games = int(sys.argv[2]) if len(sys.argv) > 2 else 12
downs, ends = Counter(), Counter()
for g in range(games):
    env = MPEnv(n_players=3, seed=10000 + g)
    P = env.reset()
    bots = [agents.make(team, seed=1000 * g + i) for i in range(3)]
    done, was_down = False, [False] * 3
    doing = [""] * 3
    while not done:
        acts, sp = {}, {}
        for i, p in enumerate(env.players):
            if not p.alive:
                continue
            a, s = bots[i].act(P[i])
            acts[i] = a
            if s:
                sp[i] = s
            b = bots[i]
            doing[i] = "rescuing" if b.rescue else "asleep" if p.sleeping else "at/near camp" if (
                b.camp and max(abs(b.camp[0] - int(p.pos[0])), abs(b.camp[1] - int(p.pos[1]))) <= 2) else \
                "walking to camp" if (b.camp and env._world.daylight < 0.75) else "no camp"
        P, info, done = env.step(acts, sp)
        light = env._world.daylight
        for i, p in enumerate(env.players):
            if p.downed and not was_down[i]:
                cause = next((e[1] for e in reversed(env.observers[i].events) if e[1].startswith("lost")), "?").split("from ")[-1]
                downs[("night" if light < 0.75 else "day", doing[i], cause)] += 1
            was_down[i] = bool(p.downed)
    ends["step %d-%d" % (env._step // 300 * 300, env._step // 300 * 300 + 299)] += 1
print("team:", team, "| games:", games)
print("game ended during:", dict(sorted(ends.items())))
print("times a player went down, by (time, what they were doing, cause):")
for k, v in downs.most_common(12):
    print("  %3d  %s" % (v, k))
