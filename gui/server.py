"""Local server for playable Multiplayer Crafter. Standard library only; binds to 127.0.0.1.

    python gui/server.py            ->  http://127.0.0.1:8765

Serves the single page (index.html) and a small JSON API around one live game (mp.session.Session).
"""
import glob
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from mp.session import Session, describe  # noqa: E402

GAME = {"session": None}
ROOT = os.path.dirname(HERE)
STEP_RE = re.compile(r"^step\s+(\d+)/(\d+) \| loss ([\d.]+) \| train acc ([\d.]+) \| ([\d.]+) tok/s \| (\d+) min left(?: \| speech loss ([\d.]+))?")
VAL_RE = re.compile(r"^\s+val acc ([\d.]+)(?: \| speech: words ([\d.]+), whole sentences ([\d.]+))?")


def training():
    """Training runs whose log (logs_train_*.txt) was written in the last 5 minutes and has not finished."""
    runs = []
    for path in sorted(glob.glob(os.path.join(ROOT, "logs_train_*.txt")), key=os.path.getmtime):
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        age = time.time() - os.path.getmtime(path)
        if age > 300 or "saved to" in text or "Traceback" in text:
            continue
        points, vals, total, left, speed = [], [], None, None, None
        for line in text.splitlines():
            m = STEP_RE.match(line)
            if m:
                total, left, speed = int(m.group(2)), int(m.group(6)), float(m.group(5))
                points.append({"step": int(m.group(1)), "loss": float(m.group(3)), "acc": float(m.group(4)),
                               "speech_loss": float(m.group(7)) if m.group(7) else None})
                continue
            v = VAL_RE.match(line)
            if v and points:
                vals.append({"step": points[-1]["step"], "acc": float(v.group(1)),
                             "words": float(v.group(2)) if v.group(2) else None,
                             "sentences": float(v.group(3)) if v.group(3) else None})
        mode = re.search(r"training mode (\S+): ([\d.]+)M trainable", text)
        runs.append({"name": os.path.basename(path)[len("logs_train_"):-4], "age_s": round(age), "total": total, "min_left": left,
                     "speed": speed, "points": points, "val": vals,
                     "mode": "%s (%sM trainable parameters)" % (mode.group(1), mode.group(2)) if mode else ""})
    return {"runs": runs, "pipeline": pipeline_status()}


STAGES = {"mp.run_baselines": ("Scoring the baseline teams", "logs_baselines.txt"),
          "mp.collect": ("Recording warm-start data from the scripted team", "logs_collect.txt"),
          "mp.train_bc": ("Training Laya", "logs_train_team_bc.txt"),
          "mp.evaluate": ("Evaluating the trained team on held-out worlds", "logs_eval_trained.txt")}


def pipeline_status():
    """What `python -m mp.run_pipeline` is doing right now, or None if it is not running."""
    path = os.path.join(ROOT, "logs_pipeline.txt")
    try:
        lines = [l.strip() for l in open(path, encoding="utf-8", errors="replace") if l.strip()]
    except OSError:
        return None
    if not lines or lines[-1].startswith("pipeline finished"):
        return None
    current = next((l for l in reversed(lines) if l.startswith(">>")), None)
    if current is None:
        return None
    module = current.split()[1]
    title, log = STAGES.get(module, (module, None))
    detail, age = "", time.time() - os.path.getmtime(path)
    if log and os.path.exists(os.path.join(ROOT, log)):
        age = min(age, time.time() - os.path.getmtime(os.path.join(ROOT, log)))
        tail = [l.strip() for l in open(os.path.join(ROOT, log), encoding="utf-8", errors="replace") if l.strip()]
        detail = next((l for l in reversed(tail) if not l.startswith(("Fetching", "Loading", "Warning"))), "")[:160]
    if age > 600:
        return None                                     # nothing has been written for 10 minutes: not running
    done = [STAGES[l.split()[1]][0] for l in lines if l.startswith(">>") and l.split()[1] in STAGES][:-1]
    return {"stage": title, "detail": detail, "done": done, "step": len(done) + 1, "of": len(STAGES)}


def new_game(body):
    kinds = [k for k in body.get("players", ["human", "scripted", "scripted"]) if k][:6]
    if not kinds or kinds.count("human") > 1:
        return {"ok": False, "error": "at most one human, at least one player"}
    if any(k not in describe()["agent_kinds"] for k in kinds):
        return {"ok": False, "error": "unknown player type"}
    if GAME["session"] is not None:
        GAME["session"].control("stop")
    GAME["session"] = Session(kinds, seed=int(body.get("seed", 0)), tick_hz=float(body.get("tick_hz", 4)),
                              mode=body.get("mode", "realtime"), hear_radius=int(body.get("hear_radius", 16)),
                              human_name=body.get("name"), give=body.get("give"))
    return {"ok": True}


def handle_post(path, body):
    s = GAME["session"]
    if path == "/api/new":
        return new_game(body)
    if s is None:
        return {"ok": False, "error": "no game running"}
    if path == "/api/act":
        s.act(body.get("action"))
    elif path == "/api/say":
        s.say(text=body.get("text"), ids=body.get("ids"))
    elif path == "/api/control":
        s.control(body.get("cmd"), body.get("value"))
    else:
        return {"ok": False, "error": "not found"}
    return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, (bytes, bytearray)):
            body = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        s = GAME["session"]
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(HERE, "index.html"), "rb") as f:
                    self.send(200, f.read(), "text/html; charset=utf-8")
            elif u.path == "/api/describe":
                self.send(200, describe())
            elif u.path == "/api/training":
                self.send(200, training())
            elif u.path == "/api/state":
                self.send(200, {"running": False} if s is None else dict(s.state(), running=True))
            elif u.path == "/api/frame":
                png = s.frame(int(q["pid"])) if s is not None else None
                self.send(200, png, "image/png") if png else self.send(404, {"error": "no frame"})
            else:
                self.send(404, {"error": "not found"})
        except (KeyError, ValueError, OSError) as e:
            self.send(400, {"error": str(e)})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.send(200, handle_post(urlparse(self.path).path, body))
        except (KeyError, ValueError, OSError) as e:
            self.send(400, {"error": str(e)})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("Multiplayer Crafter  ->  http://127.0.0.1:%d" % port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
