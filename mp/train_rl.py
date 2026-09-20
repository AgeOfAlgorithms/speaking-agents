"""Reinforcement learning on top of the warm start: a team of copies of one policy plays, and is paid as a team.

    python -m mp.train_rl --model models/team_bc --out models/team_rl
    python -m mp.train_rl --model models/team_bc --out models/team_rl_mute --no-speech     # ablation: no channel

PPO. Every player in every game is the same network; each decision (an action, or "speak" plus a sentence)
is one sample. The reward a player gets for a step is the TEAM's reward for that step (+1 the first time
anyone on the team unlocks an achievement, plus the team's mean health change), minus a little when that
player itself goes down. A sentence's log-probability is part of the decision's log-probability, so
speech is reinforced exactly as actions are -- and since speaking costs the turn, chatter has to pay.

Two additions to plain PPO, both switchable:
  --kl-anchor B   penalise drifting from the warm-start policy (actions and sentences) by B x KL. Speaking
                  costs a turn now and pays off later, for someone else; unanchored, the first thing RL
                  learns is to stop talking, and the inherited protocol decays. 0 = plain PPO.
  --critic team   centralised critic (as in MAPPO): the value estimate also reads the teammates' pooled
                  states, since the reward is the team's. `own` = each player's view only.

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
    ap.add_argument("--kl-anchor", type=float, default=0.05)
    ap.add_argument("--critic", default="team", choices=["team", "own"])
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed0", type=int, default=2000, help="training worlds; evaluation uses seeds >= 10000")
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.75)     # see --gpu-share in mp/train_bc.py

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
    print("RL from %s | %s | speech %s | starting at iteration %d | critic %s | anchor %g" % (
        source, args.train, "off" if args.no_speech else "on", start_iter, args.critic, args.kl_anchor), flush=True)
    ref = None
    if args.kl_anchor > 0:                                  # the warm start itself, frozen (not the checkpoint being resumed)
        ref_path = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)
        ref = TeamPolicy(laya.load(ref_path, device=args.device).model, tok).to(dev)
        assert ref.load_speech(ref_path)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    central = args.critic == "team"

    def team_context(cls, keys):
        """cls [R, d]: pooled vector of every player deciding this step; keys: their (game, player).
        -> for each, the mean over the OTHER players of the same game (zeros when alone)."""
        ctx = torch.zeros_like(cls)
        for r, (g, _) in enumerate(keys):
            mates = [q for q, (g2, _) in enumerate(keys) if g2 == g and q != r]
            if mates:
                ctx[r] = cls[mates].mean(0)
        return ctx

    enc = [p for k, p in policy.named_parameters() if k.startswith("core.encoder.") and p.requires_grad]
    val = list(policy.value_head.parameters()) + list(policy.value_team.parameters())
    rest = [p for k, p in policy.named_parameters() if not k.startswith(("core.encoder.", "value_head.", "value_team.")) and p.requires_grad]
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
            step_trs, step_cls, step_own = [], [], []
            for c in range(0, len(rows), 48):
                chunk, locs = rows[c:c + 48], where[c:c + 48]
                ids, att, mpos, mmask = collate(chunk)
                with torch.no_grad(), torch.autocast(**amp):
                    logits, h, memory = policy.encode(ids, att, mpos, mmask)
                    step_own.append(policy.value(h))
                    step_cls.append(h[:, 0].float())
                    if ref is not None:
                        rlogits, _, rmemory = ref.encode(ids, att, mpos, mmask)
                        ref_logp = torch.log_softmax(rlogits.float(), -1).cpu()
                dist = torch.distributions.Categorical(logits=logits)
                choice = dist.sample()
                logp = dist.log_prob(choice)
                speakers = [k for k, r in enumerate(chunk) if r["offered"][int(choice[k])] == prompts.SPEAK]
                tokens, ref_speech = {}, {}
                if speakers:
                    ntok, nmask = policy.name_tokens([chunk[k]["names"] for k in speakers], dev)
                    words, slogp, drawn = policy.speak(memory[speakers], att[speakers], ntok, nmask, sample=True, return_tokens=True)
                    if ref is not None:                     # what the warm start would have said, word by word
                        tk = torch.zeros((len(speakers), max(len(d) for d in drawn)), dtype=torch.long, device=dev)
                        for j, d in enumerate(drawn):
                            tk[j, :len(d)] = torch.tensor(d)
                        with torch.no_grad():
                            ref_words = torch.log_softmax(ref.speech_logits(rmemory[speakers], att[speakers], ntok, nmask, tk), -1).cpu()
                    for j, k in enumerate(speakers):
                        tokens[k] = drawn[j]
                        if ref is not None:
                            ref_speech[k] = ref_words[j, :len(drawn[j])]
                        logp[k] = logp[k] + slogp[j]
                        text = policy.render(words[j], chunk[k]["names"])
                        if text:
                            says[locs[k][0]][locs[k][1]] = text
                        spoken += 1
                for k, (r, (g, i)) in enumerate(zip(chunk, locs)):
                    a = r["offered"][int(choice[k])]
                    acts[g][i] = "noop" if a == prompts.SPEAK else a
                    tr = {"ids": r["ids"], "markers": r["markers"], "names": r["names"], "choice": int(choice[k]), "tokens": tokens.get(k),
                          "logp": float(logp[k]), "value": 0.0, "reward": 0.0, "done": False, "next": None, "key": (g, i),
                          "ref": ref_logp[k, :len(r["markers"])] if ref is not None else None, "ref_speech": ref_speech.get(k)}
                    step_trs.append(tr)
                    prev = games[g]["pending"][i]
                    if prev is not None:
                        prev["next"] = tr
                    games[g]["pending"][i] = tr
                    buf.append(tr)
                    decisions += 1
            if step_trs:                                    # values, once everyone deciding this step has been encoded
                cls = torch.cat(step_cls)
                ctx = team_context(cls, [tr["key"] for tr in step_trs])
                with torch.no_grad():
                    values = policy.value_team(torch.cat([cls, ctx], -1)).squeeze(-1) if central else torch.cat(step_own)
                for tr, v, x in zip(step_trs, values, ctx):
                    tr["value"] = float(v)
                    tr["ctx"] = x.half().cpu() if central else None
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
        tail_cls, tail_own = [], []
        for c in range(0, len(tails), 48):
            ids, att, mpos, mmask = collate([s for _, s in tails[c:c + 48]])
            with torch.no_grad(), torch.autocast(**amp):
                h = policy.encode(ids, att, mpos, mmask)[1]
                tail_own.append(policy.value(h))
                tail_cls.append(h[:, 0].float())
        if tails:
            cls = torch.cat(tail_cls)
            with torch.no_grad():
                v = (policy.value_team(torch.cat([cls, team_context(cls, [tr["key"] for tr, _ in tails])], -1)).squeeze(-1)
                     if central else torch.cat(tail_own))
            for (tr, _), x in zip(tails, v):
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
        kls, vlosses, ents, anchors, stopped = [], [], [], [], False
        opt.zero_grad(set_to_none=True)
        for b in range(0, len(order), args.bs):
            idx = order[b:b + args.bs]
            batch = [buf[j] for j in idx]
            ids, att, mpos, mmask = collate(batch)
            with torch.autocast(**amp):
                logits, h, memory = policy.encode(ids, att, mpos, mmask)
            dist = torch.distributions.Categorical(logits=logits)
            choice = torch.tensor([t["choice"] for t in batch], device=dev)
            logp = dist.log_prob(choice)
            drift = logits.new_zeros(len(batch))            # KL(policy || warm start), per decision
            if ref is not None:
                new = torch.log_softmax(logits, -1)
                old_ref = torch.zeros_like(new)
                for k, t in enumerate(batch):
                    old_ref[k, :len(t["ref"])] = t["ref"].to(dev)
                drift = (new.exp() * (new - old_ref)).masked_fill(~mmask, 0.0).sum(-1)
            spk = [k for k, t in enumerate(batch) if t["tokens"]]
            if spk:
                T = max(len(batch[k]["tokens"]) for k in spk)
                tk = torch.full((len(spk), T), -100, dtype=torch.long, device=dev)
                for r, k in enumerate(spk):
                    tk[r, :len(batch[k]["tokens"])] = torch.tensor(batch[k]["tokens"])
                ntok, nmask = policy.name_tokens([batch[k]["names"] for k in spk], dev)
                said = tk != -100
                words = torch.log_softmax(policy.speech_logits(memory[spk], att[spk], ntok, nmask, tk.clamp(min=0)), -1)
                extra = (words.gather(-1, tk.clamp(min=0)[..., None]).squeeze(-1) * said).sum(-1)
                where_spk = torch.tensor(spk, device=dev)
                logp = logp.index_add(0, where_spk, extra)
                if ref is not None:
                    ref_words = torch.zeros_like(words)
                    for r, k in enumerate(spk):
                        ref_words[r, :len(batch[k]["ref_speech"])] = batch[k]["ref_speech"].to(dev)
                    # a word that cannot be said (an unused name slot) has probability 0 under both
                    per_word = (words.exp() * (words - ref_words)).masked_fill(words < -1e3, 0.0).sum(-1)
                    drift = drift.index_add(0, where_spk, (per_word * said).sum(-1))
            old = torch.tensor([t["logp"] for t in batch], device=dev)
            a = torch.tensor(adv[idx], device=dev)
            ratio = torch.exp(logp - old)
            pg = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
            ret = torch.tensor([t["ret"] for t in batch], device=dev, dtype=torch.float32)
            team = torch.stack([t["ctx"] for t in batch]).to(dev) if central else None
            vloss = F.smooth_l1_loss(policy.value(h, team), ret)
            ent = dist.entropy().mean()
            ((pg + 0.5 * vloss - args.entropy * ent + args.kl_anchor * drift.mean()) / args.accum).backward()
            anchors.append(float(drift.mean().detach()))
            kls.append(float((old - logp).mean().detach()))
            vlosses.append(float(vloss.detach()))
            ents.append(float(ent.detach()))
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
              "value loss %.3f | entropy %.2f | drift from warm start %.3f | %d decisions | %.1f min/iter (play %.0fs) | %.0f min left" % (
                  it + 1, args.iters, float(np.mean([t["reward"] for t in buf])), ach, length, 100.0 * spoken / max(1, decisions),
                  float(np.mean(kls)), " (stopped early)" if stopped else "", float(np.mean(vlosses)), float(np.mean(ents)),
                  float(np.mean(anchors)), len(buf), minutes, play_s, left), flush=True)
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
