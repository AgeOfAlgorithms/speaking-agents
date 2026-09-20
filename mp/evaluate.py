"""Score a TEAM on Multiplayer Crafter. Every team plays the same held-out worlds (seeds >= 10000).

    python -m mp.evaluate --team scripted --games 50
    python -m mp.evaluate --team laya --model convaiinnovations/laya --tag laya_zeroshot --games 20

What is measured
  team score        Crafter's score computed on the TEAM: geometric mean over the 22 achievements of the
                    share of games in which anyone on the team unlocked it.
  team achievements how many of the 22 the team unlocked per game (each counts once, like the reward)
  survival          steps until nobody is left standing; and the average player's lifetime
  cooperation       revives, times downed, items handed over (dropped and picked up by someone else is
                    not tracked separately: drops and pick-ups are both reported), drinks from a
                    teammate's bucket, campfires
  talk              sentences per 100 player-steps (each one cost its speaker a turn), and the share of
                    them that any teammate could hear
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np
from crafter import constants

from core import crafter_score
from mp import agents
from mp.engine import MPEnv

ACH = list(constants.achievements)


def make_team(team, n, game, args):
    if team in ("laya", "von", "needle", "trained"):
        return None
    return [agents.make(team, seed=1000 * game + i) for i in range(n)]


def run(team, games, n_players, args, policy=None, log_every=100):
    envs = [MPEnv(n_players=n_players, seed=10000 + g, hear_radius=args.hear_radius, length=args.length) for g in range(games)]
    percepts = [e.reset() for e in envs]
    ctrls = [make_team(team, n_players, g, args) for g in range(games)]
    alive_steps = [[0] * n_players for _ in range(games)]
    said, heard_by_someone, player_steps = 0, 0, 0
    words = Counter()
    active, t, t0 = list(range(games)), 0, time.time()
    while active:
        if policy is not None:
            flat = [(g, i) for g in active for i in range(n_players) if envs[g].players[i].alive]
            outs = policy.act_batch([percepts[g][i] for g, i in flat])
            decided = {gi: o for gi, o in zip(flat, outs)}
        still = []
        for g in active:
            env, actions, speech = envs[g], {}, {}
            for i, p in enumerate(env.players):
                if not p.alive:
                    continue
                a, s = decided[(g, i)][:2] if policy is not None else ctrls[g][i].act(percepts[g][i])
                actions[i] = "noop" if p.downed else a
                if s and not args.mute:
                    speech[i] = s
                alive_steps[g][i] += 1
                player_steps += 1
            percepts[g], info, done = env.step(actions, speech)
            for pid, text in info["speech"].items():
                said += 1
                words.update(text.split())
                sp = env.players[pid]
                heard_by_someone += any(q is not sp and q.alive and env.can_hear(q.pos, sp.pos) for q in env.players)
            if not done:
                still.append(g)
        active = still
        t += 1
        if log_every and t % log_every == 0:
            print("  step %d | %d/%d games running | %.0fs" % (t, len(active), games, time.time() - t0), flush=True)

    per_game = []
    for g, env in enumerate(envs):
        stats = Counter()
        for p in env.players:
            stats.update(p.stats)
        per_game.append({"seed": 10000 + g, "names": env.names, "steps": env._step, "team_achievements": sorted(env.team_unlocked),
                         "individual": [len(u) for u in env._unlocked], "lifetimes": alive_steps[g], "stats": dict(stats)})
    rates = {a: 100.0 * sum(a in pg["team_achievements"] for pg in per_game) / games for a in ACH}
    coop = Counter()
    for pg in per_game:
        coop.update(pg["stats"])
    return {
        "team": team, "games": games, "players": n_players,
        "team_score": round(crafter_score(list(rates.values())), 2),
        "team_achievements": round(float(np.mean([len(pg["team_achievements"]) for pg in per_game])), 2),
        "individual_achievements": round(float(np.mean([np.mean(pg["individual"]) for pg in per_game])), 2),
        "game_length": round(float(np.mean([pg["steps"] for pg in per_game])), 1),
        "player_lifetime": round(float(np.mean([np.mean(pg["lifetimes"]) for pg in per_game])), 1),
        "per_game": {k: round(v / games, 2) for k, v in sorted(coop.items())},
        "sentences_per_100_player_steps": round(100.0 * said / max(1, player_steps), 2),
        "share_of_sentences_heard": round(heard_by_someone / said, 2) if said else None,
        "top_words": words.most_common(12),
        "success_rates": {k: round(v, 1) for k, v in rates.items()},
        "seconds": round(time.time() - t0, 1), "games_detail": per_game,
    }


def report(tag, r):
    print("\n=== %s ===" % tag)
    print("team score %.2f | team achievements/game %.2f | per player %.2f | game length %.0f | player lifetime %.0f | %.0fs" % (
        r["team_score"], r["team_achievements"], r["individual_achievements"], r["game_length"], r["player_lifetime"], r["seconds"]))
    print("cooperation per game:", r["per_game"])
    print("talk: %.2f sentences / 100 player-steps | heard by a teammate: %s | top words: %s" % (
        r["sentences_per_100_player_steps"], r["share_of_sentences_heard"], [w for w, _ in r["top_words"]]))
    print("unlocked by the team (% of games):", {k: v for k, v in r["success_rates"].items() if v > 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, choices=["random", "scripted", "scripted_silent", "laya", "von", "needle", "trained"])
    ap.add_argument("--model", default=None, help="hub id or local folder (default: the public checkpoint)")
    ap.add_argument("--players", type=int, default=3)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--length", type=int, default=10000)
    ap.add_argument("--hear-radius", type=int, default=16)
    ap.add_argument("--argmax", action="store_true", help="laya: greedy instead of sampling")
    ap.add_argument("--no-mask", action="store_true", help="laya: offer all 34 actions, not only the possible ones")
    ap.add_argument("--no-speech", action="store_true", help="laya / von: do not offer 'speak' as an option")
    ap.add_argument("--mute", action="store_true", help="drop everything the team says (ablation)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    policy = None
    if args.team in ("laya", "von"):
        from mp.laya_policy import LayaMP, VonMP
        backend, default = (LayaMP, "convaiinnovations/laya") if args.team == "laya" else (VonMP, "wfzyx/von-1.0")
        policy = backend(args.model or default, sample=not args.argmax, mask=not args.no_mask, can_speak=not args.no_speech)
    elif args.team == "needle":
        from mp.laya_policy import NeedleMP
        policy = NeedleMP(mask=not args.no_mask, can_speak=not args.no_speech)
    elif args.team == "trained":
        from mp.laya_policy import TrainedMP
        policy = TrainedMP(args.model, sample=not args.argmax, mask=not args.no_mask, can_speak=not args.no_speech)
    r = run(args.team, args.games, args.players, args, policy)
    r["config"] = vars(args)
    if policy is not None:
        r["truncated_inputs"] = "%d / %d" % (policy.truncated, policy.calls)
        if hasattr(policy, "refusals"):
            r["refused_to_call"] = "%d / %d" % (policy.refusals, policy.calls)
    tag = args.tag or args.team
    report(tag, r)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", tag + ".json"), "w") as f:
        json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
