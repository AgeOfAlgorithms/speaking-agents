"""Second phase of the warm start: teach the speech decoder to finish its sentences.

Only ~3% of recorded decisions are speech, so after one pass over the data the decoder has seen each
sentence once: it gets the first word and then stops ("help", instead of "help me"). This phase trains
ONLY the decoder, on every spoken sample, for many epochs. The encoder is frozen, so each sample goes
through it once and the epochs themselves take seconds. train_bc runs it automatically at the end; it can also be
applied to an existing model:

    python -m mp.tune_speech --model models/team_bc
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from mp import prompts
from mp.model import SLOW_TO_MOVE, TeamPolicy


def tune(policy, agent, rows, val_rows, epochs=25, bs=16, lr=1e-3, max_len=1280, head_max_len=640, log=print):
    from laya.common import build_sequence
    tok, dev = agent.tok, agent.device

    def prepare(rs):
        out = []
        for r in rs:
            q = agent._to_internal(prompts.question_from_offered(r["name"], r["offered"], r["use"]))
            seq, _ = build_sequence(tok, r["state"], q, max_len, head_max_len)
            words = [i for i in (TeamPolicy.word_index(w, r["names"]) for w in r["words"]) if i is not None] + [0]
            out.append((seq, words, r["names"]))
        return out

    def encode(items):
        """The encoder is frozen here, so every sample is read ONCE and its hidden states kept (fp16, in RAM)."""
        out = []
        for i in range(0, len(items), 16):
            part = items[i:i + 16]
            L = max(len(it[0]) for it in part)
            ids = torch.full((len(part), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(part), L), dtype=torch.long)
            for r, (seq, _, _) in enumerate(part):
                ids[r, :len(seq)] = torch.tensor(seq)
                att[r, :len(seq)] = 1
            ids, att = ids.to(dev), att.to(dev)
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = policy.core.encoder(input_ids=ids, attention_mask=att).last_hidden_state
                h = h + policy.core.type_emb(torch.zeros(len(part), dtype=torch.long, device=dev))[:, None, :]
                for layer in policy.core.head.layers:
                    h = layer(h, src_key_padding_mask=~att.bool())
            out += [(h[r, :len(seq)].half().cpu(), words, names) for r, (seq, words, names) in enumerate(part)]
        return out

    def batch(items):
        L, T = max(len(it[0]) for it in items), max(len(it[1]) for it in items)
        h = torch.zeros((len(items), L, items[0][0].size(-1)))
        att = torch.zeros((len(items), L), dtype=torch.long)
        tgt = torch.full((len(items), T), -100, dtype=torch.long)
        for i, (hi, words, _) in enumerate(items):
            h[i, :len(hi)] = hi.float()
            att[i, :len(hi)] = 1
            tgt[i, :len(words)] = torch.tensor(words)
        ntok, nmask = policy.name_tokens([it[2] for it in items], dev)
        return h.to(dev), att.to(dev), ntok, nmask, tgt.to(dev)

    def score(items):
        ok = total = whole = 0
        with torch.no_grad():
            for i in range(0, len(items), 32):
                h, att, ntok, nmask, tgt = batch(items[i:i + 32])
                pred = policy.speech_logits(h, att, ntok, nmask, tgt).argmax(-1)
                m = tgt != -100
                ok += int(((pred == tgt) & m).sum())
                total += int(m.sum())
                whole += int(((pred == tgt) | ~m).all(-1).sum())
        return ok / max(1, total), whole / max(1, len(items))

    policy.eval()                                               # the encoder is only read here
    t0 = time.time()
    train, val = encode(prepare(rows)), encode(prepare(val_rows))
    log("read %d spoken samples through the encoder in %.0fs" % (len(train) + len(val), time.time() - t0))
    named = [(k, p) for k, p in policy.named_parameters() if not k.startswith(("core.", "value_head."))]
    params = [p for _, p in named]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": [p for k, p in named if k not in SLOW_TO_MOVE], "lr": lr},
                             {"params": [p for k, p in named if k in SLOW_TO_MOVE], "lr": 0.02}], weight_decay=0.0)
    steps = epochs * int(np.ceil(len(train) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, 0.02], total_steps=steps)
    w0, s0 = score(val)
    log("speech tuning: %d spoken samples, %d held out | before: words %.3f, whole sentences %.3f" % (len(train), len(val), w0, s0))
    rng, t0 = np.random.RandomState(0), time.time()
    for ep in range(epochs):
        order = rng.permutation(len(train))
        for i in range(0, len(order), bs):
            h, att, ntok, nmask, tgt = batch([train[j] for j in order[i:i + bs]])
            logits = policy.speech_logits(h, att, ntok, nmask, tgt)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
        w, s = score(val)
        log("  speech epoch %d/%d | loss %.3f | held-out words %.3f, whole sentences %.3f | %.0fs" % (ep + 1, epochs, float(loss.detach()), w, s, time.time() - t0))
    with torch.no_grad():                                       # what it actually says, left to itself
        h, att, ntok, nmask, tgt = batch(val[:12])
        said, _ = policy.speak(h, att, ntok, nmask, sample=False)
        for (_, words, names), out in zip(val[:12], said):
            log("    should say: %-36s says: %s" % (TeamPolicy.render(words[:-1], names), TeamPolicy.render(out, names)))
    return w, s


def speech_rows(data_dir, val_from=57000, val_to=60000):
    rows = [json.loads(line) for line in open(os.path.join(data_dir, "samples.jsonl"), encoding="utf-8")]
    spoken = [(i, r) for i, r in enumerate(rows) if r["words"]]
    return [r for i, r in spoken if not val_from <= i < val_to], [r for i, r in spoken if val_from <= i < val_to]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/team_bc")
    ap.add_argument("--data", default="data/warmstart")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="use only this many spoken samples (quick test)")
    args = ap.parse_args()
    import laya
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = args.model if os.path.isabs(args.model) else os.path.join(root, args.model)
    agent = laya.load(path, device=args.device)
    policy = TeamPolicy(agent.model, agent.tok).to(agent.device)
    assert policy.load_speech(path)
    train, val = speech_rows(args.data)
    if args.limit:
        train, val = train[:args.limit], val[:max(8, args.limit // 4)]
    cfg = agent.cfg
    w, s = tune(policy, agent, train, val, epochs=args.epochs, max_len=cfg.get("max_len", 1280), head_max_len=cfg.get("head_max_len", 640))
    policy.save(path, cfg, {"speech_tuned": {"held_out_word_acc": w, "held_out_sentence_acc": s}})
    print("saved to", path)


if __name__ == "__main__":
    main()
