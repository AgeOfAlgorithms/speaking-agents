"""The trainable team player: Laya's encoder and option scorer, plus a small speech decoder.

One encoder pass per decision:

    [CLS] question [SEP] [MASK] opt0 [MASK] opt1 ... [MASK] speak [SEP] text state [SEP]
                          |__ option scores (Laya's own head) -> which action, or "speak"
    hidden states ______________ attended by a GRU decoder -> up to 8 words, then <end>

Speech is free grammar over a closed vocabulary. A word's score is a dot product with that word's
embedding in the encoder's own token-embedding table, so
  * words start out meaning what they mean in English (the pretrained prior under test), and
  * anything with an embedding is sayable -- including this game's randomly drawn player names.
The decoder is ~5M parameters; the expensive part (the encoder pass) is shared with the action choice.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from mp import language

MAX_WORDS = 8
EXTRA_WORDS = ["yes", "no", "not", "and", "to", "at", "my", "your", "base", "night", "safe", "danger", "where", "bring",
               "back", "stay", "now", "more", "thanks"]
WORDS = ["<end>"] + sorted({w for s in language.SLOTS for phrase in language.VOCAB[s] for w in phrase.split()} | set(EXTRA_WORDS))
WORD_ID = {w: i for i, w in enumerate(WORDS)}
MAX_NAMES = 6


class TeamPolicy(nn.Module):
    def __init__(self, core, tok, dec_dim=512):
        super().__init__()
        self.core, self.tok = core, tok
        d = core.encoder.config.hidden_size
        ids = [tok(" " + w, add_special_tokens=False)["input_ids"] for w in WORDS]
        width = max(len(i) for i in ids)
        self.register_buffer("word_tok", torch.tensor([i + [0] * (width - len(i)) for i in ids]), persistent=False)
        self.register_buffer("word_tok_mask", torch.tensor([[1.0] * len(i) + [0.0] * (width - len(i)) for i in ids]), persistent=False)
        self.end_emb = nn.Parameter(torch.randn(d) * 0.02)      # <end> and <start> are not English words
        self.start_emb = nn.Parameter(torch.randn(d) * 0.02)
        self.init_h = nn.Linear(d, dec_dim)
        self.inp = nn.Linear(d, dec_dim)
        self.query = nn.Linear(dec_dim, d)
        self.gru = nn.GRUCell(dec_dim + d, dec_dim)
        self.out = nn.Sequential(nn.Linear(dec_dim + d, d), nn.GELU(), nn.LayerNorm(d))
        self.word_bias = nn.Parameter(torch.zeros(len(WORDS)))
        self.scale = nn.Parameter(torch.tensor(10.0))
        # state value, for RL (mp/train_rl.py). The pooled vector is large and un-normalised, so it is
        # normalised first; without that one update sends the value estimates into the thousands.
        self.value_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)

    def value(self, h):
        return self.value_head(h[:, 0].float()).squeeze(-1)

    # -- shared encoder pass ---------------------------------------------------------------------
    def encode(self, input_ids, attention_mask, marker_pos, marker_mask):
        core = self.core
        h = core.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + core.type_emb(torch.zeros(h.size(0), dtype=torch.long, device=h.device))[:, None, :]
        pad = ~attention_mask.bool()
        for layer in core.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        m = torch.gather(h, 1, marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1)))
        logits = core.scorer(m).squeeze(-1).float().masked_fill(~marker_mask, -1e4)
        return logits, h

    # -- vocabulary: fixed words + this game's names -------------------------------------------------
    def _embed_table(self):
        return self.core.encoder.get_input_embeddings().weight

    def vocabulary(self, name_tok, name_tok_mask):
        """name_tok [B, MAX_NAMES, T] -> per-sample word embeddings [B, V + MAX_NAMES, d] and a validity mask."""
        E = self._embed_table()
        fixed = (E[self.word_tok] * self.word_tok_mask[..., None]).sum(1) / self.word_tok_mask.sum(1, keepdim=True)
        fixed = torch.cat([self.end_emb[None].to(fixed.dtype), fixed[1:]], 0)
        nm = name_tok_mask.to(E.dtype)
        names = (E[name_tok] * nm[..., None]).sum(2) / nm.sum(2, keepdim=True).clamp(min=1)
        table = torch.cat([fixed[None].expand(name_tok.size(0), -1, -1), names], 1)
        valid = torch.cat([torch.ones(name_tok.size(0), len(WORDS), dtype=torch.bool, device=E.device), name_tok_mask.sum(2) > 0], 1)
        return table, valid

    def _step(self, prev_emb, state, h, pad, table, valid):
        q = self.query(state)
        att = torch.einsum("bd,bld->bl", q, h.to(q.dtype)) / h.size(-1) ** 0.5
        ctx = torch.einsum("bl,bld->bd", att.masked_fill(pad, -1e4).softmax(-1), h.to(q.dtype))
        state = self.gru(torch.cat([self.inp(prev_emb), ctx], -1), state)
        z = self.out(torch.cat([state, ctx], -1))
        logits = self.scale * torch.einsum("bd,bvd->bv", F.normalize(z, dim=-1), F.normalize(table.to(z.dtype), dim=-1))
        logits = logits + F.pad(self.word_bias, (0, table.size(1) - len(WORDS)))
        return logits.float().masked_fill(~valid, -1e4), state

    def speech_logits(self, h, attention_mask, name_tok, name_tok_mask, targets):
        """Teacher-forced. targets [B, T] are indices into (WORDS + names), ending in 0 = <end>. -> [B, T, V+N]"""
        table, valid = self.vocabulary(name_tok, name_tok_mask)
        pad = ~attention_mask.bool()
        state = torch.tanh(self.init_h(h[:, 0].float()))
        prev = self.start_emb[None].expand(h.size(0), -1).float()
        out = []
        for t in range(targets.size(1)):
            logits, state = self._step(prev, state, h.float(), pad, table.float(), valid)
            out.append(logits)
            prev = table.float()[torch.arange(h.size(0)), targets[:, t].clamp(min=0)]
        return torch.stack(out, 1)

    def sentence_logp(self, h, attention_mask, name_tok, name_tok_mask, tokens):
        """Log-probability of each sampled sentence. tokens [B, T]: word indices exactly as `speak` drew them
        (including the <end> it stopped on, if it did), padded with -100."""
        logits = self.speech_logits(h, attention_mask, name_tok, name_tok_mask, tokens.clamp(min=0))
        logp = torch.log_softmax(logits, -1).gather(-1, tokens.clamp(min=0)[..., None]).squeeze(-1)
        return (logp * (tokens != -100)).sum(-1)

    @torch.no_grad()
    def speak(self, h, attention_mask, name_tok, name_tok_mask, sample=True, temperature=1.0, return_tokens=False):
        """-> list of word-index lists (without <end>), and the summed log-probability of each sentence.
        With return_tokens, also the exact token lists drawn (with <end>), which sentence_logp can re-score."""
        table, valid = self.vocabulary(name_tok, name_tok_mask)
        pad = ~attention_mask.bool()
        b = h.size(0)
        state = torch.tanh(self.init_h(h[:, 0].float()))
        prev = self.start_emb[None].expand(b, -1).float()
        alive = torch.ones(b, dtype=torch.bool, device=h.device)
        words, logp = [[] for _ in range(b)], torch.zeros(b, device=h.device)
        drawn = [[] for _ in range(b)]
        for _ in range(MAX_WORDS + 1):
            logits, state = self._step(prev, state, h.float(), pad, table.float(), valid)
            dist = torch.distributions.Categorical(logits=logits / temperature)
            w = dist.sample() if sample else logits.argmax(-1)
            logp = logp + dist.log_prob(w) * alive
            for i in range(b):
                if alive[i]:
                    drawn[i].append(int(w[i]))
                if alive[i] and w[i] != 0 and len(words[i]) < MAX_WORDS:
                    words[i].append(int(w[i]))
            alive = alive & (w != 0)
            if not alive.any():
                break
            prev = table.float()[torch.arange(b), w]
        return (words, logp, drawn) if return_tokens else (words, logp)

    # -- names ---------------------------------------------------------------------------------------
    def name_tokens(self, names_per_sample, device):
        toks = [[self.tok(" " + n, add_special_tokens=False)["input_ids"][:4] for n in names[:MAX_NAMES]] for names in names_per_sample]
        width = max([len(t) for ts in toks for t in ts] + [1])
        ids = torch.zeros(len(toks), MAX_NAMES, width, dtype=torch.long)
        mask = torch.zeros(len(toks), MAX_NAMES, width)
        for i, ts in enumerate(toks):
            for j, t in enumerate(ts):
                ids[i, j, :len(t)] = torch.tensor(t)
                mask[i, j, :len(t)] = 1
        return ids.to(device), mask.to(device)

    @staticmethod
    def word_index(word, names):
        """Index of a word in (WORDS + this sample's names), or None if it cannot be said."""
        if word in WORD_ID:
            return WORD_ID[word]
        return len(WORDS) + names.index(word) if word in names[:MAX_NAMES] else None

    @staticmethod
    def render(indices, names):
        return " ".join(WORDS[i] if i < len(WORDS) else names[i - len(WORDS)] for i in indices)

    # -- checkpoints (Laya's own layout + speech.safetensors, so laya.load() still opens the core) ----
    def save(self, out, cfg, extra=None):
        os.makedirs(out, exist_ok=True)
        save_file({k: v.half().contiguous().cpu() for k, v in self.core.state_dict().items()}, os.path.join(out, "model.safetensors"))
        speech = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items() if not k.startswith("core.")}
        save_file(speech, os.path.join(out, "speech.safetensors"))
        self.core.encoder.config.save_pretrained(os.path.join(out, "encoder"))
        self.tok.save_pretrained(os.path.join(out, "tokenizer"))
        cfg = dict(cfg, fine_tuned=True, model_name="laya-team-player", words=WORDS, temperature=[1.0, 1.0, 1.0],
                   temperature_by_options={}, **(extra or {}))
        with open(os.path.join(out, "rl_agent_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    def load_speech(self, path):
        f = os.path.join(path, "speech.safetensors")
        if os.path.exists(f):
            self.load_state_dict(load_file(f), strict=False)
            return True
        return False
