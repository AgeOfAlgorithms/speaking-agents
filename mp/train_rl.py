"""Reinforcement learning on top of the warm start: a team of copies of one policy plays, and is paid as a team.

    python -m mp.train_rl --model models/team_bc --out models/team_rl
    python -m mp.train_rl --model models/team_bc --out models/team_rl_mute --no-speech     # ablation: no channel

PPO. Every player in every game is the same network; each decision (an action, or "speak" plus a sentence)
is one sample. The reward a player gets for a step is the TEAM's reward for that step (+1 the first time
anyone on the team unlocks an achievement, plus the team's mean health change), minus a little when that
player itself goes down. A sentence's log-probability is part of the decision's log-probability, so
speech is reinforced exactly as actions are -- and since speaking costs the turn, chatter has to pay.

Only the top encoder layers train by default: it halves the cost of the backward pass, which is what
makes this feasible on one GPU. A checkpoint is written every iteration; --resume continues from it.
"""
import argparse
import json
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from mp import prompts
from mp.engine import MPEnv, state_dict
from mp.model import TeamPolicy
from mp.train_bc import set_trainable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/team_bc")
    ap.add_argument("--out", default="models/team_rl")
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--players", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=128, help="environment steps per game per iteration")
    ap.add_argument("--train", default="top8")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=3e-5)
    ap.add_argument("--lr-value", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--max-kl", type=float, default=0.05, help="stop an update early once the policy has moved this far")
    ap.add_argument("--down-penalty", type=float, default=0.5)
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed0", type=int, default=2000, help="training worlds; evaluation uses seeds >= 10000")
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)      # see --gpu-share in mp/train_bc.py

    import laya
    from laya.common import build_sequence
    torch.manual_seed(0)
    rng = np.random.RandomState(0)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = os.path.join(args.out, "checkpoint")
    start_iter, source = 0, args.model
    if args.resume and os.path.exists(os.path.join(ckpt, "progress.json")):
        start_iter, source = json.load(open(os.path.join(ckpt, "progress.json")))["iter"], ckpt
    path = source if os.path.isabs(source) else os.path.join(root, source)
    agent = laya.load(path, device=args.device)
    tok, dev, cfg = agent.tok, agent.device, agent.cfg
    max_len, head_max_len = cfg.get("max_len", 1280), cfg.get("head_max_len", 640)
    policy = TeamPolicy(agent.model, tok).to(dev)
    assert policy.load_speech(path), "no speech head in %s" % path
    set_trainable(policy, args.train)
    policy.eval()                                           # no dropout: old and new log-probabilities must be comparable
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")
    print("RL from %s | %s | speech %s | starting at iteration %d" % (source, args.train, "off" if args.no_speech else "on", start_iter), flush=True)

    enc = [p for k, p in policy.named_parameters() if k.startswith("core.encoder.") and p.requires_grad]
    val = list(policy.value_head.parameters())
    rest = [p for k, p in policy.named_parameters() if not k.startswith(("core.encoder.", "value_head.")) and p.requires_grad]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr_encoder}, {"params": rest, "lr": args.lr_head},
                             {"params": val, "lr": args.lr_value}], weight_decay=0.0)

    def collate(rows):
        L, K = max(len(r["ids"]) for r in rows), max(len(r["markers"]) for r in rows)
        ids = torch.full((len(rows), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(rows), L), dtype=torch.long)
        mpos = torch.zeros((len(rows), K), dtype=torch.long)
        mmask = torch.zeros((len(rows), K), dtype=torch.bool)
        for i, r in enumerate(rows):
            ids[i, :len(r["ids"])] = torch.tensor(r["ids"])
            att[i, :len(r["ids"])] = 1
            mpos[i, :len(r["markers"])] = torch.tensor(r["markers"])
            mmask[i, :len(r["markers"])] = True
        return ids.to(dev), att.to(dev), mpos.to(dev), mmask.to(dev)

    def sequence(P):
        q, offered = prompts.action_question(P, True, not args.no_speech)
        ids, markers = build_sequence(tok, json.dumps(state_dict(P), separators=(",", ":")), agent._to_internal(q), max_len, head_max_len)
        return {"ids": ids, "markers": markers, "offered": offered, "names": [P["name"]] + P["teammates"]}

    next_seed = [args.seed0 + start_iter * args.games]

    def new_game():
        env = MPEnv(n_players=args.players, seed=next_seed[0])
        next_seed[0] += 1
        return {"env": env, "P": env.reset(), "pending": [None] * args.players, "was_down": [False] * args.players}

    games = [new_game() for _ in range(args.games)]
    finished = deque(maxlen=40)                             # recent completed games: the learning curve
    t_start = time.time()

    for it in range(start_iter, args.iters):
        t0 = time.time()
        buf, spoken, decisions = [], 0, 0
        # ---------------------------------------------------------------- play
        for _ in range(args.horizon):
            rows, where = [], []
            for g, G in enumerate(games):
                for i, p in enumerate(G["env"].players):
                    if p.alive and not p.sleeping:
                        rows.append(sequence(G["P"][i]))
                        where.append((g, i))
            acts = [dict() for _ in games]
            says = [dict() for _ in games]
            for c in range(0, len(rows), 48):
                chunk, locs = rows[c:c + 48], where[c:c + 48]
                ids, att, mpos, mmask = collate(chunk)
                with torch.no_grad(), torch.autocast(**amp):
                    logits, h = policy.encode(ids, att, mpos, mmask)
                    values = policy.value(h)
                dist = torch.distributions.Categorical(logits=logits)
                choice = dist.sample()
                logp = dist.log_prob(choice)
                speakers = [k for k, r in enumerate(chunk) if r["offered"][int(choice[k])] == prompts.SPEAK]
                tokens = {}
                if speakers:
                    ntok, nmask = policy.name_tokens([chunk[k]["names"] for k in speakers], dev)
                    words, slogp, drawn = policy.speak(h[speakers], att[speakers], ntok, nmask, sample=True, return_tokens=True)
                    for j, k in enumerate(speakers):
                        tokens[k] = drawn[j]
                        logp[k] = logp[k] + slogp[j]
                        text = policy.render(words[j], chunk[k]["names"])
                        if text:
                            says[locs[k][0]][locs[k][1]] = text
                        spoken += 1
                for k, (r, (g, i)) in enumerate(zip(chunk, locs)):
                    a = r["offered"][int(choice[k])]
                    acts[g][i] = "noop" if a == prompts.SPEAK else a
                    tr = {"ids": r["ids"], "markers": r["markers"], "names": r["names"], "choice": int(choice[k]), "tokens": tokens.get(k),
                          "logp": float(logp[k]), "value": float(values[k]), "reward": 0.0, "done": False, "next": None, "key": (g, i)}
                    prev = games[g]["pending"][i]
                    if prev is not None:
                        prev["next"] = tr
                    games[g]["pending"][i] = tr
                    buf.append(tr)
                    decisions += 1
            for g, G in enumerate(games):
                env = G["env"]
                G["P"], info, done = env.step(acts[g], says[g])
                for i, p in enumerate(env.players):
                    tr = G["pending"][i]
                    if tr is None:
                        continue
                    tr["reward"] += info["team_reward"]
                    if p.downed and not G["was_down"][i]:
                        tr["reward"] -= args.down_penalty
                    G["was_down"][i] = bool(p.downed)
                    if done or not p.alive:
                        tr["done"] = True
                        G["pending"][i] = None
                if done:
                    stats = {k: sum(q.stats[k] for q in env.players) for k in env.players[0].stats}
                    finished.append({"ach": len(env.team_unlocked), "len": env._step, **stats})
                    games[g] = new_game()
        # values of the states the rollout stopped in, to bootstrap unfinished trajectories
        tails = [(G["pending"][i], sequence(G["P"][i])) for G in games for i, p in enumerate(G["env"].players)
                 if G["pending"][i] is not None and p.alive]
        for c in range(0, len(tails), 48):
            ids, att, mpos, mmask = collate([s for _, s in tails[c:c + 48]])
            with torch.no_grad(), torch.autocast(**amp):
                v = policy.value(policy.encode(ids, att, mpos, mmask)[1])
            for (tr, _), x in zip(tails[c:c + 48], v):
                tr["bootstrap"] = float(x)
        for tr in reversed(buf):                            # GAE, walking each player's chain backwards
            nxt = tr["next"]
            if tr["done"]:
                next_v, next_adv = 0.0, 0.0
            elif nxt is None:
                next_v, next_adv = tr.get("bootstrap", 0.0), 0.0
            else:
                next_v, next_adv = nxt["value"], nxt["adv"]
            delta = tr["reward"] + args.gamma * next_v - tr["value"]
            tr["adv"] = delta + args.gamma * args.lam * next_adv
            tr["ret"] = tr["adv"] + tr["value"]
        play_s = time.time() - t0
        # ---------------------------------------------------------------- learn
        adv = np.array([t["adv"] for t in buf], dtype=np.float32)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        order = rng.permutation(len(buf))
        kls, vlosses, ents, stopped = [], [], [], False
        opt.zero_grad(set_to_none=True)
        for b in range(0, len(order), args.bs):
            idx = order[b:b + args.bs]
            batch = [buf[j] for j in idx]
            ids, att, mpos, mmask = collate(batch)
            with torch.autocast(**amp):
                logits, h = policy.encode(ids, att, mpos, mmask)
            dist = torch.distributions.Categorical(logits=logits)
            choice = torch.tensor([t["choice"] for t in batch], device=dev)
            logp = dist.log_prob(choice)
            spk = [k for k, t in enumerate(batch) if t["tokens"]]
            if spk:
                T = max(len(batch[k]["tokens"]) for k in spk)
                tk = torch.full((len(spk), T), -100, dtype=torch.long, device=dev)
                for r, k in enumerate(spk):
                    tk[r, :len(batch[k]["tokens"])] = torch.tensor(batch[k]["tokens"])
                ntok, nmask = policy.name_tokens([batch[k]["names"] for k in spk], dev)
                extra = policy.sentence_logp(h[spk], att[spk], ntok, nmask, tk)
                logp = logp.index_add(0, torch.tensor(spk, device=dev), extra)
            old = torch.tensor([t["logp"] for t in batch], device=dev)
            a = torch.tensor(adv[idx], device=dev)
            ratio = torch.exp(logp - old)
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            ret = torch.tensor([t["ret"] for t in batch], device=dev, dtype=torch.float32)
            vloss = F.smooth_l1_loss(policy.value(h), ret)
            ent = dist.entropy().mean()
            ((pg + 0.5 * vloss - args.entropy * ent) / args.accum).backward()
            kls.append(float((old - logp).mean().detach()))
            vlosses.append(float(vloss))
            ents.append(float(ent))
            if (b // args.bs + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if np.mean(kls[-args.accum * 4:]) > args.max_kl:
                    stopped = True
                    break
        # ---------------------------------------------------------------- report + save
        recent = list(finished)
        ach = np.mean([f["ach"] for f in recent]) if recent else 0.0      # 0 until the first game of the run ends
        length = np.mean([f["len"] for f in recent]) if recent else 0.0
        minutes = (time.time() - t0) / 60
        left = (args.iters - it - 1) * (time.time() - t_start) / (it - start_iter + 1) / 60
        print("iter %3d/%d | reward/decision %.4f | team achievements %.2f | game length %.0f | speaks %.2f%% | kl %.4f%s | "
              "value loss %.3f | entropy %.2f | %d decisions | %.1f min/iter (play %.0fs) | %.0f min left" % (
                  it + 1, args.iters, float(np.mean([t["reward"] for t in buf])), ach, length, 100.0 * spoken / max(1, decisions),
                  float(np.mean(kls)), " (stopped early)" if stopped else "", float(np.mean(vlosses)), float(np.mean(ents)),
                  len(buf), minutes, play_s, left), flush=True)
        policy.save(ckpt, cfg, {"max_len": max_len, "head_max_len": head_max_len})
        with open(os.path.join(ckpt, "progress.json"), "w") as f:
            json.dump({"iter": it + 1, "of": args.iters, "args": vars(args),
                       "recent": {"team_achievements": float(ach), "game_length": float(length)}}, f)

    policy.save(args.out, cfg, {"max_len": max_len, "head_max_len": head_max_len, "rl": vars(args)})
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        json.dump({"iters": args.iters, "hours": (time.time() - t_start) / 3600, "args": vars(args)}, f, indent=1)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
