"""Teach the speech decoder to speak before anyone has to learn to listen.

Learning to talk and to understand at the same time is a chicken-and-egg problem: a message is worth
nothing until someone acts on it, and nobody learns to act on noise. So the two are separated:

  1. HERE (supervised, no game running): for thousands of recorded states, the decoder learns to say
     things that are TRUE of the state it is looking at (mp/describe.py): what is where, how many, what it
     has and lacks, what it needs, who is nearby, what time it is. This is where the compass words, the
     distances and the quantities get attached to the world.
  2. then a short mixed phase so it still says what the scripted team says at the moments they say it
     ("help me", "come campfire here"), without forgetting 1.
  3. WHEN speaking is worth a turn, and what to do about what one hears, is left to reinforcement learning.

The encoder is frozen and only read. Progress is measured by TRUTHFULNESS: of the sentences the decoder
produces for held-out states, how many are true of that state.

    python -m mp.pretrain_speech --model models/team_bc
"""
import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from mp import prompts
from mp.describe import all_true, true_sentences
from mp.model import SLOW_TO_MOVE, TeamPolicy
from mp.tune_speech import tune


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/team_bc")
    ap.add_argument("--data", default="data/warmstart")
    ap.add_argument("--states", type=int, default=16000)
    ap.add_argument("--per-state", type=int, default=4, help="different true sentences taught per state per pass")
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--mix-epochs", type=int, default=8)
    ap.add_argument("--mix-described", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--memory-layer", type=int, default=None, help="encoder layer the decoder reads (default: mp.model.MEMORY_LAYER; "
                    "0 = the final states after Laya's head)")
    ap.add_argument("--val-states", type=int, default=400)
    ap.add_argument("--probe-steps", type=int, default=0, help="stop after this many steps, skip the mixed phase, save nothing")
    ap.add_argument("--gpu-share", type=float, default=0.16, help="this job is small; leave the card to whatever else is training")
    args = ap.parse_args()
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_share)

    import laya
    from laya.common import build_sequence
    random.seed(0)
    torch.manual_seed(0)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)
    agent = laya.load(path, device=args.device)
    tok, dev, cfg = agent.tok, agent.device, agent.cfg
    max_len, head_max_len = cfg.get("max_len", 1280), cfg.get("head_max_len", 640)
    policy = TeamPolicy(agent.model, tok).to(dev)               # a fresh decoder
    policy.eval()
    if args.memory_layer is not None:
        policy.memory_layer = args.memory_layer
    if dev.type == "cuda":
        policy.core.encoder.to(torch.bfloat16)                  # only read here; halves this job's share of the card
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    rows = [json.loads(line) for line in open(os.path.join(args.data, "samples.jsonl"), encoding="utf-8")]
    held_out = range(57000, 60000)                              # the same games every other script holds out
    pool = [i for i in range(len(rows)) if i not in held_out]
    random.shuffle(pool)
    train_ids, val_ids = pool[:args.states], random.sample(list(held_out), args.val_states)
    spoken = [(i, r) for i, r in enumerate(rows) if r["words"]]

    def encode(batch_rows):
        seqs = [build_sequence(tok, r["state"], agent._to_internal(prompts.question_from_offered(r["name"], r["offered"], r["use"])),
                               max_len, head_max_len)[0] for r in batch_rows]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s)
            att[k, :len(s)] = 1
        ids, att = ids.to(dev), att.to(dev)
        with torch.no_grad(), torch.autocast(**amp):
            return policy.hidden(ids, att)[1].float(), att

    def targets(r, k):
        """k true sentences for this state, drawn family-first so rare kinds (danger, teammates) get their share."""
        fams = true_sentences(r["state"], r["offered"])
        picked = []
        for fam in random.sample(list(fams), min(k, len(fams))):
            picked.append(random.choice(fams[fam]))
        return picked

    def truthfulness(sample):
        ok = total = 0
        kinds = {}
        examples = []
        for c in range(0, len(val_ids), 16):
            rs = [rows[i] for i in val_ids[c:c + 16]]
            h, att = encode(rs)
            ntok, nmask = policy.name_tokens([r["names"] for r in rs], dev)
            said, _ = policy.speak(h, att, ntok, nmask, sample=sample)
            for r, words in zip(rs, said):
                text = TeamPolicy.render(words, r["names"])
                true = text in all_true(r["state"], r["offered"])
                ok += true
                total += 1
                kinds[text.split(" ")[0] if text else ""] = kinds.get(text.split(" ")[0] if text else "", 0) + 1
                if len(examples) < 10:
                    examples.append(("TRUE  " if true else "false ") + text)
        return ok / total, examples

    named = [(k, p) for k, p in policy.named_parameters() if not k.startswith(("core.", "value_head.", "value_team."))]
    params = [p for _, p in named]
    opt = torch.optim.AdamW([{"params": [p for k, p in named if k not in SLOW_TO_MOVE], "lr": args.lr},
                             {"params": [p for k, p in named if k in SLOW_TO_MOVE], "lr": 5e-3}], weight_decay=0.0)
    per_pass = int(np.ceil(len(train_ids) / args.bs))
    steps = args.probe_steps or per_pass * args.passes
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, 5e-3], total_steps=steps, pct_start=0.1)
    print("speech pretraining: %d states x %d true sentences each x %d passes, %d steps" % (len(train_ids), args.per_state, args.passes, steps), flush=True)
    t0, run = time.time(), []
    for step in range(steps):
        at = step % per_pass
        if at == 0:
            random.shuffle(train_ids)
        rs = [rows[i] for i in train_ids[at * args.bs:(at + 1) * args.bs]]
        h, att = encode(rs)
        which, sents, names = [], [], []
        for k, r in enumerate(rs):
            for words in targets(r, args.per_state):
                which.append(k)
                sents.append([TeamPolicy.word_index(w, r["names"]) for w in words] + [0])
                names.append(r["names"])
        if not which:
            sched.step()
            continue
        T = max(len(s) for s in sents)
        tgt = torch.full((len(sents), T), -100, dtype=torch.long, device=dev)
        for k, s in enumerate(sents):
            tgt[k, :len(s)] = torch.tensor(s)
        idx = torch.tensor(which, device=dev)
        ntok, nmask = policy.name_tokens(names, dev)
        logits = policy.speech_logits(h, att, ntok, nmask, tgt, rows=idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        run.append(float(loss.detach()))
        if (step + 1) % 100 == 0 or step + 1 == steps:
            el = time.time() - t0
            line = "step %5d/%d | loss %.3f | %.0f min left" % (step + 1, steps, np.mean(run), (steps - step - 1) * el / (step + 1) / 60)
            run = []
            if (step + 1) % 300 == 0 or step + 1 == steps:
                greedy, _ = truthfulness(sample=False)
                sampled, ex = truthfulness(sample=True)
                line += " | held-out sentences that are TRUE of the state: %.0f%% (most likely sentence), %.0f%% (sampled)" % (100 * greedy, 100 * sampled)
            print(line, flush=True)
    for e in ex:
        print("    " + e, flush=True)
    if args.probe_steps:
        return

    # 2. mixed phase: the scripted team's sentences at the scripted team's moments, plus described states so 1 is kept
    described = []
    for i in train_ids[:args.mix_described]:
        t = targets(rows[i], 1)
        if t:
            described.append(dict(rows[i], words=t[0]))
    train_spoken = [r for i, r in spoken if i not in held_out]
    val_spoken = [r for i, r in spoken if i in held_out]
    print("mixed phase: %d scripted sentences + %d described states" % (len(train_spoken), len(described)), flush=True)
    wacc, sacc = tune(policy, agent, train_spoken + described, val_spoken, epochs=args.mix_epochs, lr=2e-4, slow_lr=1e-3,
                      max_len=max_len, head_max_len=head_max_len)
    greedy, _ = truthfulness(sample=False)
    sampled, ex = truthfulness(sample=True)
    print("after the mixed phase | scripted moments: words %.3f, whole sentences %.3f | any state: true %.0f%% (most likely), %.0f%% (sampled)" % (
        wacc, sacc, 100 * greedy, 100 * sampled), flush=True)
    for e in ex:
        print("    " + e, flush=True)
    backup = os.path.join(path, "speech_before_pretraining.safetensors")
    if not os.path.exists(backup):
        shutil.copy(os.path.join(path, "speech.safetensors"), backup)
    policy.core.encoder.float()
    policy.save_speech(path)
    with open(os.path.join(path, "speech_report.json"), "w") as f:
        json.dump({"scripted_word_acc": wacc, "scripted_sentence_acc": sacc, "true_most_likely": greedy, "true_sampled": sampled,
                   "args": vars(args)}, f, indent=1)
    print("saved the decoder to", path)


if __name__ == "__main__":
    main()
