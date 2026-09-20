"""Trace the scripted team through its first evening and night. python -m mp.trace_night [seed]"""
import sys
from mp import agents
from mp.agents import LIT
from mp.engine import MPEnv

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10001
env = MPEnv(n_players=3, seed=seed); P = env.reset()
bots = [agents.make("scripted", seed=i) for i in range(3)]
done = False
while not done and env._step < 320:
    acts, sp = {}, {}
    for i, p in enumerate(env.players):
        if p.alive:
            a, s = bots[i].act(P[i]); acts[i] = a
            if s: sp[i] = s
    if env._step >= 90 and env._step % 15 == 0:
        row = []
        for i, p in enumerate(env.players):
            b = bots[i]
            d = max(abs(b.camp[0] - int(p.pos[0])), abs(b.camp[1] - int(p.pos[1]))) if b.camp else None
            fires = [(f.kind[:5], f.fuel) for f in env.fires if max(abs(int(f.pos[0]) - int(p.pos[0])), abs(int(f.pos[1]) - int(p.pos[1]))) <= 4]
            state = "DEAD" if not p.alive else "DOWN" if p.downed else "zzz" if p.sleeping else acts.get(i, "-")
            row.append("%-4s hp%d d%d w%d s%d b%s camp=%s fires%s %s" % (p.name[:4], p.health, p.inventory["drink"], p.inventory["wood"], p.inventory["stone"],
                                                                     sum(p.bucket_water.values()) if (p.inventory["wood_bucket"] or p.inventory["iron_bucket"]) else "-", d, fires, state))
        print("t=%3d light %.2f | " % (env._step, env._world.daylight) + " | ".join(row))
    P, info, done = env.step(acts, sp)
    for e in info["events"]: print("      >>", e, "at step", env._step)
    for pid, t in info["speech"].items(): print('      "%s" - %s' % (t, env.players[pid].name))
