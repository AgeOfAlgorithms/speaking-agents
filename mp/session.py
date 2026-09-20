"""A live multiplayer game running in a background thread, for the GUI.

One slot may be a human (actions and speech arrive over HTTP); the rest are controllers from
mp.agents. Real-time mode ticks at a fixed rate and a human who presses nothing simply stands still;
turn mode waits for the human's move, which is kinder when you are still learning the keys.
"""
import io
import json
import threading
import time

from PIL import Image

from mp import agents, language
from mp.engine import ACTIONS, BUCKETS, CAMPFIRE, MPEnv, state_dict

BUBBLE_TICKS = 14


class Session:
    def __init__(self, kinds, seed=0, tick_hz=4.0, mode="realtime", hear_radius=16, length=10000, human_name=None, give=None):
        self.kinds, self.seed, self.tick_hz, self.mode = list(kinds), int(seed), float(tick_hz), mode
        human_name = "".join(c for c in (human_name or "") if c.isalnum() or c in " -_")[:12].strip() or None
        names = [human_name if k == "human" else None for k in kinds]        # everyone else gets a random name
        self.env = MPEnv(n_players=len(kinds), names=names, seed=self.seed, hear_radius=hear_radius, length=length)
        self.percepts = self.env.reset()
        if give and "human" in self.kinds:                 # sandbox start (API only): begin with some items
            inv = self.env.players[self.kinds.index("human")].inventory
            for item, n in give.items():
                if item in inv:
                    inv[item] = int(n)
        self.controllers = [agents.make(k, seed=self.seed * 10 + i) for i, k in enumerate(self.kinds)]
        self.human = self.kinds.index("human") if "human" in self.kinds else None
        self.lock = threading.Lock()
        self.paused, self.stopped, self.done = False, False, False
        self.utterances, self.feed, self.last_actions = [], [], {}
        self.step_once = False
        self.error = None
        self._frames = {}
        self._publish()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # -- game loop -----------------------------------------------------------------------------
    def _run(self):
        try:
            while not self.stopped and not self.done:
                t0 = time.time()
                human = self.controllers[self.human] if self.human is not None else None
                waiting = (self.mode == "turn" and human is not None and self.env.players[self.human].standing
                           and human.action is None and human.speech is None)
                if (self.paused and not self.step_once) or waiting:
                    time.sleep(0.03)
                    continue
                self.step_once = False
                self._tick()
                time.sleep(max(0.0, 1.0 / self.tick_hz - (time.time() - t0)))
        except Exception as e:  # surface it in the GUI rather than dying silently in a thread
            import traceback
            self.error = "%s\n%s" % (e, traceback.format_exc())

    def _tick(self):
        actions, speech = {}, {}
        for p, ctrl, P in zip(self.env.players, self.controllers, self.percepts):
            if not p.alive:
                continue
            a, s = ctrl.act(P)
            if p.downed:
                a = "noop"
            actions[p.pid] = a
            if s:
                speech[p.pid] = s
        self.percepts, info, self.done = self.env.step(actions, speech)
        step = self.env._step
        with self.lock:
            self.last_actions = actions
            for pid, text in info["speech"].items():
                p = self.env.players[pid]
                self.utterances.append({"step": step, "pid": pid, "name": p.name, "text": text,
                                        "x": int(p.pos[0]), "y": int(p.pos[1])})
            for e in info["events"]:
                self.feed.append({"step": step, "text": e})
            for a in info["new_team_achievements"]:
                self.feed.append({"step": step, "text": "team unlocked " + a.replace("_", " ")})
            self.utterances, self.feed = self.utterances[-400:], self.feed[-200:]
        self._publish()

    def _publish(self):
        snap = self.env.snapshot()
        step = snap["step"]
        frames = {p.pid: self.env.render(p.pid) for p in self.env.players if p.alive}
        for pl, ob, P, kind, ctrl in zip(snap["players"], self.env.observers, self.percepts, self.kinds, self.controllers):
            pl["kind"] = kind
            pl["action"] = self.last_actions.get(pl["pid"])
            pl["state"] = state_dict(P)
            pl["comms"] = ob.comms[-40:]
            pl["blocked"] = P.get("blocked", {})
            pl["use"] = dict(zip(("label", "ok"), P.get("use", ("nothing", False))))
            pl["decision"] = getattr(ctrl, "last_probs", None)   # Laya players: the options it was offered and their probabilities
        with self.lock:
            snap.update(seed=self.seed, mode=self.mode, tick_hz=self.tick_hz, paused=self.paused, done=self.done,
                        human=self.human, hear_radius=self.env.hear_radius, error=self.error,
                        bubbles=[u for u in self.utterances if step - u["step"] <= BUBBLE_TICKS],
                        utterances=self.utterances[-60:], feed=self.feed[-30:])
            self.snapshot, self._raw = snap, frames
            self._frames = {}

    # -- GUI api -------------------------------------------------------------------------------
    def state(self):
        with self.lock:
            s = dict(self.snapshot)
            s["error"] = self.error
            s["paused"] = self.paused
            return s

    def frame(self, pid):
        with self.lock:
            if pid not in self._frames and pid in self._raw:
                bio = io.BytesIO()
                Image.fromarray(self._raw[pid]).save(bio, format="PNG")
                self._frames[pid] = bio.getvalue()
            return self._frames.get(pid)

    def act(self, action):
        if self.human is not None and action in ACTIONS:
            self.controllers[self.human].action = action

    def say(self, text=None, ids=None):
        if self.human is None:
            return
        if ids is not None and len(ids) == len(language.SLOTS) and all(0 <= i < n for i, n in zip(ids, language.SIZES)):
            self.controllers[self.human].speech = tuple(int(i) for i in ids)
        elif text:
            self.controllers[self.human].speech = str(text)[:120]

    def control(self, cmd, value=None):
        if cmd == "pause":
            self.paused = True
        elif cmd == "resume":
            self.paused = False
        elif cmd == "step":
            self.step_once = True
        elif cmd == "stop":
            self.stopped = True
        elif cmd == "speed" and value:
            self.tick_hz = max(0.5, min(30.0, float(value)))
        elif cmd == "mode" and value in ("realtime", "turn"):
            self.mode = value


def describe():
    """Static facts the page needs once: actions, vocabulary, rules."""
    from crafter import constants

    def cost(uses):
        return ", ".join("%d %s" % (v, k.replace("_", " ")) for k, v in uses.items())

    craft = [{"action": "make_" + k, "label": k.replace("_", " "), "cost": cost(v["uses"]),
              "needs": " + ".join(v["nearby"])} for k, v in constants.make.items()]
    place = [{"action": "place_" + k, "label": "sapling" if k == "plant" else k, "cost": cost(v["uses"])}
             for k, v in constants.place.items()]
    place += [{"action": "place_" + k, "label": "%s (burns %d steps)" % (k.replace("_", " "), v["burn"]), "cost": cost(v["cost"])}
              for k, v in CAMPFIRE["kinds"].items()]
    place += [
              {"action": "place_bucket", "label": "bucket (shared water)", "cost": "your bucket"}]
    drop = [{"action": a, "label": a[len("drop_"):].replace("_", " "), "item": a[len("drop_"):]}
            for a in ACTIONS if a.startswith("drop_")]
    import os
    trained = os.path.exists(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "team_bc", "speech.safetensors"))
    kinds = {"scripted": "Scripted (talks)", "scripted_silent": "Scripted (silent)", "laya_zeroshot": "Laya, untrained",
             "von_zeroshot": "von, untrained", "needle_zeroshot": "needle, untrained", "random": "Random", "human": "You"}
    if trained:
        kinds = {"laya_trained": "Laya, trained", **kinds}
    return {"actions": ACTIONS, "slots": list(language.SLOTS), "vocab": language.VOCAB,
            "agent_kinds": kinds,
            "agent_notes": {"scripted": "hand-written rules: plays, announces finds, calls for help, comes to revive",
                            "scripted_silent": "same hand-written rules, never speaks",
                            "laya_trained": "Laya warm-started on the scripted co-op team: one pass picks the action or 'speak', a small decoder writes the sentence",
                            "laya_zeroshot": "Laya straight from the hub, never trained on this game (loads in ~15 s)",
                            "needle_zeroshot": "Needle 3, a small tool-calling model, straight from the hub (CPU, ~1 s per decision)",
                            "von_zeroshot": "von-1.0 (NLI model) straight from the hub, never trained on this game",
                            "random": "random actions", "human": "you"},
            "menus": {"craft": craft, "place": place, "drop": drop},
            "buckets": BUCKETS, "campfire": CAMPFIRE}
