"""Warm start: teach the team player (mp/model.py) to do what the scripted co-op team does.

A checkpoint is written to <out>/checkpoint every --save-every steps; add --resume to pick an
interrupted run up from there (same data order, learning-rate schedule restored; the optimiser's
moment estimates start fresh, which costs a few dozen steps of re-warming, not the run).

    python -m mp.train_bc --out models/team_bc                       # full fine-tune from Laya
    python -m mp.train_bc --train top8 --out models/team_bc_top8     # only the top 8 encoder layers
    python -m mp.train_bc --train frozen --out models/team_bc_frozen # encoder frozen
    python -m mp.train_bc --encoder von --out models/team_bc_von     # von's encoder under the same head

Two losses on one encoder pass: which option (action or "speak"), and -- when the scripted player
spoke -- the words of its sentence. Progress lines match what the GUI's training panel reads.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mp import prompts
from mp.model import MAX_NAMES, TeamPolicy, SLOW_TO_MOVE
from mp.tune_speech import tune


def load_core(args):
    import laya
    agent = laya.load(args.model, device="cuda")
    if args.encoder == "von":
        from transformers import AutoModelForSequenceClassification
        von = AutoModelForSequenceClassification.from_pretrained("wfzyx/von-1.0", dtype=torch.float32)
        sd = {k[len("model."):]: v for k, v in von.state_dict().items() if k.startswith("model.")}
        missing, unexpected = agent.model.encoder.load_state_dict(sd, strict=False)
        print("von encoder loaded into Laya's head | missing %d, unexpected %d tensors" % (len(missing), len(unexpected)), flush=True)
    return agent


def set_trainable(policy, mode):
    enc = policy.core.encoder
    if mode == "frozen":
        for p in enc.parameters():
            p.requires_grad_(False)
    elif mode.startswith("top"):
        keep = int(mode[3:])
        for p in enc.parameters():
            p.requires_grad_(False)
        for layer in enc.layers[-keep:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in enc.final_norm.parameters():
            p.requires_grad_(True)
    n = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("training mode %s: %.1fM trainable parameters" % (mode, n / 1e6), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/warmstart")
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--encoder", default="laya", choices=["laya", "von"])
    ap.add_argument("--train", default="full", help="full | frozen | topN (e.g. top8)")
    ap.add_argument("--out", default="models/team_bc")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val", type=int, default=3000)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=4e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--lr-speech", type=float, default=5e-4)
    ap.add_argument("--max-len", type=int, default=1280)
    ap.add_argument("--head-max-len", type=int, default=640)
    ap.add_argument("--save-every", type=int, default=1000, help="write a checkpoint to <out>/checkpoint every N steps")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint if there is one")
    ap.add_argument("--gpu-share", type=float, default=0.8, help="most of the card PyTorch may hold. On Windows a process that "
                    "outgrows free VRAM is silently moved to system memory and runs ~10x slower; capped, PyTorch frees its cache instead")
    args = ap.parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_share)

    from laya.common import build_sequence
    torch.manual_seed(0)
    np.random.seed(0)
    agent = load_core(args)
    tok, dev = agent.tok, agent.device
    policy = TeamPolicy(agent.model, tok).to(dev)
    ckpt_dir, start_step = os.path.join(args.out, "checkpoint"), 0
    if args.resume and os.path.exists(os.path.join(ckpt_dir, "progress.json")):
        from safetensors.torch import load_file
        policy.core.load_state_dict({k: v.float() for k, v in load_file(os.path.join(ckpt_dir, "model.safetensors")).items()})
        policy.load_speech(ckpt_dir)
        start_step = json.load(open(os.path.join(ckpt_dir, "progress.json")))["step"]
        print("resumed from %s at step %d" % (ckpt_dir, start_step), flush=True)
    set_trainable(policy, args.train)

    rows = all_rows = [json.loads(line) for line in open(os.path.join(args.data, "samples.jsonl"), encoding="utf-8")]
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    data = []
    for r in rows:
        q = agent._to_internal(prompts.question_from_offered(r["name"], r["offered"], r["use"]))
        seq, markers = build_sequence(tok, r["state"], q, args.max_len, args.head_max_len)
        assert len(markers) == len(r["offered"])
        words = None
        if r["words"]:
            idx = [TeamPolicy.word_index(w, r["names"]) for w in r["words"]]
            words = [i for i in idx if i is not None] + [0]
        data.append((seq, markers, r["choice"], words, r["names"], r["offered"][r["choice"]]))
    lens = np.array([len(d[0]) for d in data])
    print("prepared %d samples in %.0fs | tokens/sample mean %.0f max %d | %d are 'speak'" % (
        len(data), time.time() - t0, lens.mean(), lens.max(), sum(d[3] is not None for d in data)), flush=True)

    n_train = len(data) - args.val
    labels = [d[5] for d in data[:n_train]]
    counts = {a: labels.count(a) for a in set(labels)}
    median = float(np.median(list(counts.values())))
    weight = {a: float(np.clip(np.sqrt(median / c), 1.0, 5.0)) for a, c in counts.items()}

    def batch(idx):
        items = [data[i] for i in idx]
        L, K = max(len(it[0]) for it in items), max(len(it[1]) for it in items)
        ids = torch.full((len(items), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(items), L), dtype=torch.long)
        mpos = torch.zeros((len(items), K), dtype=torch.long)
        mmask = torch.zeros((len(items), K), dtype=torch.bool)
        for r, it in enumerate(items):
            ids[r, :len(it[0])] = torch.tensor(it[0])
            att[r, :len(it[0])] = 1
            mpos[r, :len(it[1])] = torch.tensor(it[1])
            mmask[r, :len(it[1])] = True
        y = torch.tensor([it[2] for it in items])
        w = torch.tensor([weight.get(it[5], 1.0) for it in items])
        spk = [r for r, it in enumerate(items) if it[3] is not None]
        T = max([len(items[r][3]) for r in spk] + [1])
        tgt = torch.full((len(spk), T), -100, dtype=torch.long)
        for k, r in enumerate(spk):
            tgt[k, :len(items[r][3])] = torch.tensor(items[r][3])
        ntok, nmask = policy.name_tokens([items[r][4] for r in spk] or [[]], dev)
        return (ids.to(dev), att.to(dev), mpos.to(dev), mmask.to(dev), y.to(dev), w.to(dev), spk, tgt.to(dev), ntok, nmask)

    def losses(b):
        ids, att, mpos, mmask, y, w, spk, tgt, ntok, nmask = b
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, h, memory = policy.encode(ids, att, mpos, mmask)
        # label smoothing over the options that were actually offered (never onto the padding slots)
        k = mmask.sum(-1, keepdim=True).clamp(min=2).float()
        target = (mmask.float() * (0.05 / (k - 1))).scatter(1, y[:, None], 0.95)
        logp = torch.log_softmax(logits, -1).masked_fill(~mmask, 0.0)
        act_loss = (-(target * logp).sum(-1) * w).sum() / w.sum()
        sp_loss, sp_logits = logits.new_zeros(()), None
        if spk:
            sp_logits = policy.speech_logits(memory[spk], att[spk], ntok, nmask, tgt)
            sp_loss = F.cross_entropy(sp_logits.reshape(-1, sp_logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
        return act_loss, sp_loss, logits, sp_logits

    def evaluate():
        policy.eval()
        ok = total = sent_ok = sent = word_ok = word = 0
        with torch.no_grad():
            for i in range(n_train, len(data), 16):
                b = batch(list(range(i, min(i + 16, len(data)))))
                _, _, logits, sp_logits = losses(b)
                ok += int((logits.argmax(-1) == b[4]).sum())
                total += len(b[4])
                if sp_logits is not None:
                    pred, tgt = sp_logits.argmax(-1), b[7]
                    m = tgt != -100
                    word_ok += int(((pred == tgt) & m).sum())
                    word += int(m.sum())
                    sent_ok += int((((pred == tgt) | ~m).all(-1)).sum())
                    sent += len(tgt)
        policy.train()
        return ok / max(1, total), word_ok / max(1, word), sent_ok / max(1, sent), sent

    policy.train()
    enc = [p for k, p in policy.named_parameters() if k.startswith("core.encoder.") and p.requires_grad]
    head = [p for k, p in policy.named_parameters() if k.startswith("core.") and not k.startswith("core.encoder.")]
    speech = [p for k, p in policy.named_parameters() if not k.startswith("core.") and k not in SLOW_TO_MOVE]
    slow = [p for k, p in policy.named_parameters() if k in SLOW_TO_MOVE]
    groups = [{"params": head, "lr": args.lr_head}, {"params": speech, "lr": args.lr_speech},
              {"params": slow, "lr": 5e-3, "weight_decay": 0.0}]
    if enc:
        groups.append({"params": enc, "lr": args.lr_encoder})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps_per_epoch = int(np.ceil(n_train / args.bs))
    max_steps = int(steps_per_epoch * args.epochs)
    updates = max(1, max_steps // args.accum)
    warm = max(1, int(0.03 * updates))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: min(1.0, (u + 1) / warm) * (0.04 + 0.96 * 0.5 * (1 + np.cos(np.pi * min(1.0, u / updates)))))

    def train_on(idx, share=1.0):
        """Forward + backward on one micro-batch -> (action loss, speech loss, accuracy), tokens read.
        A batch of unusually long states can outgrow the memory cap: it is then done as two halves."""
        try:
            b = batch(idx)
            act_loss, sp_loss, logits, _ = losses(b)
            ((act_loss + sp_loss) * share / args.accum).backward()
            return share * np.array([float(act_loss.detach()), float(sp_loss.detach()),
                                     float((logits.argmax(-1) == b[4]).float().mean())]), int(b[1].sum())
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
        b = act_loss = sp_loss = logits = None
        torch.cuda.empty_cache()
        half = len(idx) // 2
        s1, t1 = train_on(idx[:half], share * half / len(idx))
        s2, t2 = train_on(idx[half:], share * (len(idx) - half) / len(idx))
        return s1 + s2, t1 + t2

    print("training: %d samples, %.2f epochs, %d optimiser updates" % (n_train, args.epochs, updates), flush=True)
    step, t0, run, tokens, done, seen = 0, time.time(), np.zeros(3), 0, False, 0
    for _ in range(start_step // args.accum):
        sched.step()                                   # put the learning-rate schedule where it was
    while not done:
        order = np.random.permutation(n_train)         # seeded above, so a resumed run sees the same order
        for i in range(0, n_train, args.bs):
            if step < start_step:                      # already trained on this batch before the interruption
                step += 1
                continue
            stats, n_tok = train_on(order[i:i + args.bs])
            step += 1
            tokens += n_tok
            run += stats
            seen += 1
            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if step % 100 == 0:
                el = time.time() - t0
                print("step %5d/%d | loss %.3f | train acc %.3f | %.0f tok/s | %.0f min left | speech loss %.3f" % (
                    step, max_steps, run[0] / max(1, seen), run[2] / max(1, seen), tokens / el,
                    (max_steps - step) * el / max(1, step - start_step) / 60, run[1] / max(1, seen)), flush=True)
                run[:] = 0
                seen = 0
            if step % args.save_every == 0 and step < max_steps:
                policy.eval()
                policy.save(ckpt_dir, agent.cfg, {"max_len": args.max_len, "head_max_len": args.head_max_len})
                with open(os.path.join(ckpt_dir, "progress.json"), "w") as f:
                    json.dump({"step": step, "of": max_steps, "args": vars(args)}, f)
                policy.train()
                print("  checkpoint saved at step %d" % step, flush=True)
            if step % 1000 == 0:
                acc, wacc, sacc, n = evaluate()
                print("  val acc %.3f | speech: words %.3f, whole sentences %.3f (%d spoken)" % (acc, wacc, sacc, n), flush=True)
            if step >= max_steps:
                done = True
                break

    acc, wacc, sacc, n = evaluate()
    print("final val acc %.3f | speech: words %.3f, whole sentences %.3f (%d spoken)" % (acc, wacc, sacc, n))
    policy.eval()
    policy.save(args.out, agent.cfg, {"trained_with": vars(args), "max_len": args.max_len, "head_max_len": args.head_max_len})
    # speech is ~3% of the decisions, so one pass leaves the decoder under-trained: give it its own phase
    del opt
    torch.cuda.empty_cache()
    spoken = [(i, r) for i, r in enumerate(all_rows) if r["words"]]
    wacc, sacc = tune(policy, agent, [r for i, r in spoken if not n_train <= i < len(data)], [r for i, r in spoken if n_train <= i < len(data)],
                      max_len=args.max_len, head_max_len=args.head_max_len)
    policy.save(args.out, agent.cfg, {"trained_with": vars(args), "max_len": args.max_len, "head_max_len": args.head_max_len})
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        json.dump({"val_action_acc": acc, "val_word_acc": wacc, "val_sentence_acc": sacc, "minutes": (time.time() - t0) / 60,
                   "args": vars(args)}, f, indent=1)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
