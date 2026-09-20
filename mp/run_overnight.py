"""The queue that follows the first warm start. Waits for mp.run_pipeline, then runs each stage in turn,
skipping whatever is already done, and commits + pushes the results after every stage so nothing
finished is ever only on this machine.

    python -m mp.run_overnight
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


def save(message):
    subprocess.run(["git", "add", "-A"], check=False)
    body = message + "\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
    if subprocess.run(GIT + ["commit", "-q", "-m", body], check=False).returncode == 0:
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        subprocess.run(["git", "push", "-q", "origin", "main"], env=env, check=False)
        print("   saved + pushed: %s" % message, flush=True)


def evaluate(tag, model, games="20", extra=()):
    run(["mp.evaluate", "--team", "trained", "--model", model, "--games", games, "--tag", tag, *extra],
        "logs_eval_%s.txt" % tag, skip_if="results/%s.json" % tag)


while not os.path.exists("results/laya_trained.json"):            # the first pipeline is still going
    if "pipeline finished" in open("logs_pipeline.txt", encoding="utf-8").read():
        break
    time.sleep(60)
save("Results: warm-started Laya team (full fine-tune)")

# zero-shot under the final rules (the earlier zero-shot numbers predate shades and short campfires)
run(["mp.evaluate", "--team", "laya", "--games", "20", "--tag", "laya_zeroshot"], "logs_zeroshot.txt", skip_if="results/laya_zeroshot.json")
run(["mp.evaluate", "--team", "laya", "--games", "20", "--argmax", "--tag", "laya_zeroshot_greedy"], "logs_zeroshot.txt",
    skip_if="results/laya_zeroshot_greedy.json")
run(["mp.evaluate", "--team", "von", "--games", "12", "--tag", "von_zeroshot"], "logs_zeroshot.txt", skip_if="results/von_zeroshot.json")
save("Results: zero-shot Laya and von under the final rules")

# same warm start, von's encoder under the same one-pass head: which pretraining is the better start?
run(["mp.train_bc", "--limit", "60000", "--encoder", "von", "--out", "models/team_bc_von", "--resume"], "logs_train_team_bc_von.txt",
    skip_if="models/team_bc_von/train_report.json")
evaluate("von_trained", "models/team_bc_von")
save("Results: warm start from von's encoder")

# how much does unfreezing the encoder buy?
for mode in ("top8", "frozen"):
    run(["mp.train_bc", "--limit", "60000", "--train", mode, "--out", "models/team_bc_" + mode, "--resume"],
        "logs_train_team_bc_%s.txt" % mode, skip_if="models/team_bc_%s/train_report.json" % mode)
    evaluate("laya_trained_" + mode, "models/team_bc_" + mode)
    save("Results: warm start with encoder %s" % mode)
print("overnight queue finished   [%s]" % time.strftime("%H:%M"), flush=True)
