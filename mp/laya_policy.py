"""Text decision models as multiplayer Crafter players: Laya, and von (github.com/wfzyx/von).

One `choice` question per agent per step: the instruction, the actions that are possible right now
(mp.prompts), and the agent's text state (mp.engine.state_dict -- which carries its name, teammates,
event log and everything it has heard). All agents of all parallel games go through in one batch.

Speaking takes the turn, so "speak" is simply one more option in the action list. Zero-shot, an
untrained model has no speech head, so when it picks "speak" the words are found the only way it can
do anything -- by scoring options: each of the six slots is asked as its own `choice` question (blank
is an option). The trained model replaces those six questions with a small decoder on the same pass.
"""
import json
import os

import numpy as np
import torch

from mp import language, prompts
from mp.engine import state_dict

BLANK = "(nothing)"
SPEECH_INSTRUCTIONS = (
    "You are {name}, playing a cooperative survival game. You may say one short sentence that teammates within earshot "
    "will hear: tell them what you found and where, ask for what you need, warn them, or call for help. You are choosing "
    "the {slot} of that sentence; choose {blank} to leave it out."
)


class LayaMP:
    def __init__(self, model="convaiinnovations/laya", sample=True, mask=True, can_speak=True, temperature=1.0,
                 max_len=1280, head_max_len=640, seed=0, device="cuda"):
        import laya
        from laya.common import build_sequence, collate_items
        self._build, self._collate = build_sequence, collate_items
        if not model.startswith("convaiinnovations/") and not os.path.isabs(model):
            model = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model)
        self.agent = laya.load(model, device=device)
        self.tok, self.dev = self.agent.tok, self.agent.device
        self.sample, self.mask, self.can_speak, self.temperature = sample, mask, can_speak, temperature
        self.max_len, self.head_max_len = max_len, head_max_len
        self.rng = np.random.RandomState(seed)
        self.truncated = self.calls = 0

    # -- one batched forward pass over arbitrary `choice` questions ----------------------------------
    @torch.no_grad()
    def _score(self, states, questions, chunk=48):
        out = []
        for i in range(0, len(states), chunk):
            items = []
            for state, q in zip(states[i:i + chunk], questions[i:i + chunk]):
                qi = self.agent._to_internal(q)
                seq, markers = self._build(self.tok, state, qi, self.max_len, self.head_max_len)
                assert len(markers) == len(qi["crit"]), "options did not fit in head_max_len"
                self.calls += 1
                self.truncated += len(seq) >= self.max_len
                items.append({"ids": seq, "markers": markers, "qtype": 0})
            b = self._collate([items], self.tok.pad_token_id)
            with torch.autocast(device_type=self.dev.type, dtype=self.agent.dtype, enabled=self.dev.type == "cuda"):
                logits, _ = self.agent.model(b["input_ids"].to(self.dev), b["attention_mask"].to(self.dev),
                                             b["marker_pos"].to(self.dev), b["marker_mask"].to(self.dev), b["qtype"].to(self.dev))
            logits = logits.float().cpu().numpy()
            for r, it in enumerate(items):
                z = logits[r, :len(it["markers"])] / self.temperature
                p = np.exp(z - z.max())
                out.append(p / p.sum())
        return out

    def _pick(self, p):
        return int(self.rng.choice(len(p), p=p)) if self.sample else int(p.argmax())

    # -- acting --------------------------------------------------------------------------------------
    def act_batch(self, percepts):
        """percepts: multiplayer percepts (any games, any players). -> [(action, speech_ids or None, info)]"""
        live = [i for i, P in enumerate(percepts) if not P["downed"] and not P["sleeping"]]
        states = {i: json.dumps(state_dict(percepts[i]), separators=(",", ":")) for i in live}
        asked = {i: prompts.action_question(percepts[i], self.mask, self.can_speak) for i in live}
        probs = self._score([states[i] for i in live], [asked[i][0] for i in live])
        out = [("noop", None, None)] * len(percepts)
        for i, p in zip(live, probs):
            offered = asked[i][1]
            k = self._pick(p)
            out[i] = (offered[k], None, {"options": [prompts.label(a) for a in offered], "probs": [round(float(v), 4) for v in p],
                                         "chose": prompts.label(offered[k])})

        speakers = [i for i in live if out[i][0] == prompts.SPEAK]
        for i in speakers:
            out[i] = ("noop", None, out[i][2])
        if speakers:
            qs, ss = [], []
            for i in speakers:
                for slot in language.SLOTS:
                    words = [w or BLANK for w in language.VOCAB[slot]]
                    qs.append({"type": "choice", "criteria": words,
                               "instructions": SPEECH_INSTRUCTIONS.format(name=percepts[i]["name"], slot=slot, blank=BLANK)})
                    ss.append(states[i])
            sp = self._score(ss, qs)
            n = len(language.SLOTS)
            for j, i in enumerate(speakers):
                ids = tuple(self._pick(p) for p in sp[j * n:(j + 1) * n])
                if any(ids):
                    out[i] = (out[i][0], ids, out[i][2])
        return out


class VonMP(LayaMP):
    """von-1.0 (wfzyx/von) behind the same interface, asked the very same questions as Laya.

    von is ModernBERT-large fine-tuned for natural-language inference. It decides by entailment: for
    each option it asks "given the state, does '<instructions> <option description>' follow?" and
    soft-maxes the entailment logits over the options (src/von/backends/berta_backend.py). So where
    Laya reads one sequence holding all K options, von reads K sequences -- about K times the compute.
    Its SDK truncates at 512 tokens, which would cut our state, so the model is called directly here
    with a longer limit; everything else (hypothesis format, state as indented JSON, calibration
    temperature) follows its own code."""

    def __init__(self, model="wfzyx/von-1.0", sample=True, mask=True, can_speak=True, temperature=None,
                 max_len=1024, seed=0, device="cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model, dtype=torch.bfloat16).to(self.dev).eval()
        labels = {str(v).lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = next((i for name, i in labels.items() if "entail" in name), 0)
        if temperature is None:
            temperature = 1.0
            try:
                from huggingface_hub import hf_hub_download
                with open(hf_hub_download(model, "calibration.json"), encoding="utf-8") as f:
                    temperature = float(json.load(f).get("temperature", 1.0))
            except Exception:
                pass
        self.sample, self.mask, self.can_speak, self.temperature, self.max_len = sample, mask, can_speak, temperature, max_len
        self.rng = np.random.RandomState(seed)
        self.truncated = self.calls = 0

    @torch.no_grad()
    def _score(self, states, questions, chunk=64):
        premises, hypotheses, spans = [], [], []
        for state, q in zip(states, questions):
            crit = q["criteria"]
            opts = list(crit.items()) if isinstance(crit, dict) else [(c, None) for c in crit]
            spans.append((len(premises), len(opts)))
            premise = json.dumps(json.loads(state), indent=2, ensure_ascii=False)       # von's own state formatting
            for name, desc in opts:
                premises.append(premise)
                hypotheses.append("%s %s" % (q["instructions"], desc if desc else name))
        if not premises:            # every agent asleep or down this step: nothing to score
            return []
        scores = []
        for i in range(0, len(premises), chunk):
            enc = self.tok(premises[i:i + chunk], hypotheses[i:i + chunk], padding=True, truncation="only_first",
                           max_length=self.max_len, return_tensors="pt").to(self.dev)
            self.calls += enc["input_ids"].shape[0]
            self.truncated += int((enc["attention_mask"].sum(-1) >= self.max_len).sum())
            scores.append(self.model(**enc).logits[:, self.entail].float().cpu().numpy())
        scores = np.concatenate(scores)
        out = []
        for start, k in spans:
            z = scores[start:start + k] / self.temperature
            p = np.exp(z - z.max())
            out.append(p / p.sum())
        return out


class TrainedMP(LayaMP):
    """A trained team player (mp/model.py): one encoder pass gives the action; if the action is "speak",
    the small decoder on that same pass writes the sentence (free grammar, up to 8 words, names allowed)."""

    def __init__(self, model, sample=True, mask=True, can_speak=True, temperature=1.0, seed=0, device="cuda", **kw):
        super().__init__(model, sample=sample, mask=mask, can_speak=can_speak, temperature=temperature, seed=seed, device=device,
                         max_len=kw.get("max_len", 1280), head_max_len=kw.get("head_max_len", 640))
        from mp.model import TeamPolicy
        self.policy = TeamPolicy(self.agent.model, self.tok).to(self.dev).eval()
        path = model if os.path.isabs(model) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model)
        assert self.policy.load_speech(path), "no speech.safetensors in %s" % path
        torch.manual_seed(seed)

    @torch.no_grad()
    def act_batch(self, percepts, chunk=48):
        out = [("noop", None, None)] * len(percepts)
        live = [i for i, P in enumerate(percepts) if not P["sleeping"]]      # a downed player may still call for help
        for c in range(0, len(live), chunk):
            idx = live[c:c + chunk]
            items, offered = [], []
            for i in idx:
                q, off = prompts.action_question(percepts[i], self.mask, self.can_speak)
                seq, markers = self._build(self.tok, json.dumps(state_dict(percepts[i]), separators=(",", ":")),
                                           self.agent._to_internal(q), self.max_len, self.head_max_len)
                items.append({"ids": seq, "markers": markers, "qtype": 0})
                offered.append(off)
            b = self._collate([items], self.tok.pad_token_id)
            ids, att = b["input_ids"].to(self.dev), b["attention_mask"].to(self.dev)
            with torch.autocast(device_type=self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                logits, _, memory = self.policy.encode(ids, att, b["marker_pos"].to(self.dev), b["marker_mask"].to(self.dev))
            probs = torch.softmax(logits / self.temperature, -1).cpu().numpy()
            speakers = []
            for r, i in enumerate(idx):
                p = probs[r, :len(offered[r])]
                p = p / p.sum()
                k = self._pick(p)
                info = {"options": [prompts.label(a) for a in offered[r]], "probs": [round(float(v), 4) for v in p],
                        "chose": prompts.label(offered[r][k])}
                if offered[r][k] == prompts.SPEAK:
                    speakers.append((r, i, info))
                    out[i] = ("noop", None, info)
                else:
                    out[i] = (offered[r][k], None, info)
            if speakers:
                rows = [r for r, _, _ in speakers]
                names = [[percepts[i]["name"]] + percepts[i]["teammates"] for _, i, _ in speakers]
                ntok, nmask = self.policy.name_tokens(names, self.dev)
                words, _ = self.policy.speak(memory[rows], att[rows], ntok, nmask, sample=self.sample)
                for (r, i, info), w, nm in zip(speakers, words, names):
                    text = self.policy.render(w, nm)
                    info["said"] = text
                    out[i] = ("noop", text or None, info)
        return out


_SHARED = {}


class LayaAgent:
    """Controller for the live game: several Laya players in one game share one loaded model."""

    def __init__(self, model="convaiinnovations/laya", kind="laya_zeroshot", backend=LayaMP, **kw):
        key = (model, tuple(sorted(kw.items())))
        if key not in _SHARED:
            _SHARED[key] = backend(model, **kw)
        self.policy, self.kind = _SHARED[key], kind
        self.last_probs = None

    def act(self, P):
        action, speech, info = self.policy.act_batch([P])[0]
        self.last_probs = info
        return action, speech
