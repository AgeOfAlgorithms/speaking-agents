"""After the local-reward RL pair: the von-encoder comparison, which twice ran out of GPU memory at the usual batch size.
Same effective batch (4 x 8 instead of 8 x 4), half the activation memory. Resumable.

    python -m mp.run_von_queue
"""
import os
import subprocess
import sys
import time

PY = [sys.executable, "-u", "-m"]
GIT = ["git", "-c", "user.name=seannam", "-c", "user.email=seannam@protonmail.com"]


def run(args, log, skip_if=None):
    if skip_if and os.path.exists(skip_if):
        print("skip (have %s)" % skip_if, flush=True)
        return
    print(">> %s   [%s]" % (" ".join(args), time.strftime("%H:%M")), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        subprocess.run(PY + args, stdout=f, stderr=subprocess.STDOUT, check=False)


while "local reward queue finished" not in open("logs_local_reward_queue.txt", encoding="utf-8").read():
    time.sleep(60)
run(["mp.train_bc", "--limit", "60000", "--encoder", "von", "--bs", "4", "--accum", "8", "--out", "models/team_bc_von", "--resume"],
    "logs_train_team_bc_von.txt", skip_if="models/team_bc_von/train_report.json")
run(["mp.pretrain_speech", "--model", "models/team_bc_von"], "logs_pretrain_speech_team_bc_von.txt", skip_if="models/team_bc_von/speech_report.json")
run(["mp.evaluate", "--team", "trained", "--model", "models/team_bc_von", "--games", "20", "--tag", "von_trained"], "logs_eval_von_trained.txt",
    skip_if="results/von_trained.json")
subprocess.run(["git", "add", "-A"], check=False)
body = "Results: von's encoder under Laya's head, warm-started the same way\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
if subprocess.run(GIT + ["commit", "-q", "-m", body], check=False).returncode == 0:
    subprocess.run(["git", "push", "-q", "origin", "main"], env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), check=False)
print("von queue finished   [%s]" % time.strftime("%H:%M"), flush=True)
