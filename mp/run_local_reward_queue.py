"""The RL pair (speech on / speech off) under the LOCAL reward scheme
(mp/train_rl.py --reward local: own health + the health of teammates in view, a small bonus per step
standing and per step within earshot of a teammate). Resumable; commits and pushes after each run.

    python -m mp.run_local_reward_queue [iterations]
"""
import os
import subprocess
import sys
import time

PY = [sys.executable, "-u", "-m"]
GIT = ["git", "-c", "user.name=seannam", "-c", "user.email=seannam@protonmail.com"]
iters = sys.argv[1] if len(sys.argv) > 1 else "60"


def run(args, log, skip_if=None):
    if skip_if and os.path.exists(skip_if):
        print("skip (have %s)" % skip_if, flush=True)
        return
    print(">> %s   [%s]" % (" ".join(args), time.strftime("%H:%M")), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        subprocess.run(PY + args, stdout=f, stderr=subprocess.STDOUT, check=False)


def save(message):
    subprocess.run(["git", "add", "-A"], check=False)
    body = message + "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
    if subprocess.run(GIT + ["commit", "-q", "-m", body], check=False).returncode == 0:
        subprocess.run(["git", "push", "-q", "origin", "main"], env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), check=False)
        print("   saved + pushed: %s" % message, flush=True)


for out, tag, extra in (("models/team_rl_local", "laya_rl_local", []), ("models/team_rl_local_mute", "laya_rl_local_mute", ["--no-speech"])):
    run(["mp.train_rl", "--model", "models/team_bc", "--out", out, "--iters", iters, "--reward", "local", "--resume"] + extra,
        "logs_train_%s.txt" % os.path.basename(out).replace("team_", ""), skip_if=out + "/train_report.json")
    run(["mp.evaluate", "--team", "trained", "--model", out, "--games", "20", "--tag", tag] + extra, "logs_eval_%s.txt" % tag,
        skip_if="results/%s.json" % tag)
    save("Results: RL under the local reward scheme, speech %s" % ("off" if extra else "on"))
print("local reward queue finished   [%s]" % time.strftime("%H:%M"), flush=True)
