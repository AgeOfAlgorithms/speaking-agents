"""Record warm-start data from the scripted co-op team: for every decision, exactly what a learned player
would be given and what the scripted player did.

    python -m mp.collect --samples 80000

Each line of data/warmstart/samples.jsonl:
  state     the text state (mp.engine.state_dict), as the JSON string the model reads
  name      who is deciding;  names: everyone alive (these become sayable words)
  offered   the engine actions possible at that moment, plus "speak"
  use       what `use` would do right now (it is part of the option text)
  choice    index into `offered` of what the scripted player did
  words     the sentence, as a list of words, when the choice was "speak"

--noise: with this probability a random possible action is EXECUTED while the scripted choice is what
gets RECORDED, so the learner also sees states a little off the scripted path, with the right answer.
"""
import argparse
import json
import os
import time
from collections import Counter

import numpy as np

from mp import agents, language, prompts
from mp.engine import ACTIONS, MPEnv, state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=80000)
    ap.add_argument("--players", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--keep-asleep", type=float, default=0.1, help="share of asleep / downed-and-silent steps to keep")
    ap.add_argument("--out", default="data/warmstart")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(0)
    n, seed, t0 = 0, 0, time.time()
    choices, spoken, lengths = Counter(), Counter(), []
    with open(os.path.join(args.out, "samples.jsonl"), "w", encoding="utf-8") as f:
        while n < args.samples:
            assert seed < 10000, "ran into the evaluation worlds"
            env = MPEnv(n_players=args.players, seed=seed)
            P = env.reset()
            bots = [agents.make("scripted", seed=7 * seed + i) for i in range(args.players)]
            done = False
            while not done and n < args.samples:
                acts, speech = {}, {}
                for i, p in enumerate(env.players):
                    if not p.alive:
                        continue
                    a, s = bots[i].act(P[i])
                    words = language.render(s).split() if s else None
                    _, offered = prompts.action_question(P[i])
                    label = prompts.SPEAK if words else (a if a in offered else "noop")
                    dull = (P[i]["sleeping"] or P[i]["downed"]) and not words
                    if label in offered and (not dull or rng.rand() < args.keep_asleep):
                        f.write(json.dumps({"state": json.dumps(state_dict(P[i]), separators=(",", ":")), "name": P[i]["name"],
                                            "names": [P[i]["name"]] + P[i]["teammates"], "offered": offered, "use": P[i]["use"][0],
                                            "choice": offered.index(label), "words": words}) + "\n")
                        choices[label] += 1
                        if words:
                            spoken[" ".join(words[:3])] += 1
                        n += 1
                    if words:
                        speech[i] = s
                    elif rng.rand() < args.noise and not p.downed and not p.sleeping:
                        possible = [x for x in offered if x not in (prompts.SPEAK, "sleep", "noop") and not x.startswith(("drop_", "place_"))]
                        a = possible[rng.randint(len(possible))] if possible else a
                    acts[i] = a
                P, info, done = env.step(acts, speech)
            lengths.append(env._step)
            seed += 1
            if seed % 10 == 0:
                print("  %d games, %d / %d samples, %.0fs" % (seed, n, args.samples, time.time() - t0), flush=True)
    info = {"samples": n, "games": seed, "mean_game_length": float(np.mean(lengths)), "noise": args.noise,
            "choices": dict(choices.most_common()), "sentence_openings": dict(spoken.most_common(12))}
    with open(os.path.join(args.out, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
