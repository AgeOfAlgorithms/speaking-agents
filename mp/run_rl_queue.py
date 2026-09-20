"""After the warm-start comparisons (mp.run_overnight): reinforcement learning from the warm-started Laya,
then evaluation on the held-out worlds. Resumable; commits and pushes when each part finishes.

    python -m mp.run_rl_queue [iterations]
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


while "overnight queue finished" not in open("logs_overnight.txt", encoding="utf-8").read():
    time.sleep(60)                                           # the GPU is still busy with the warm-start comparisons

# the von-encoder warm start ran out of GPU memory in the overnight queue; train_bc now halves such batches
run(["mp.train_bc", "--limit", "60000", "--encoder", "von", "--out", "models/team_bc_von", "--resume"], "logs_train_team_bc_von.txt",
    skip_if="models/team_bc_von/train_report.json")
# speak first, understand later: the decoders learn to say true things about recorded states (mp/pretrain_speech.py)
# BEFORE any RL, so that RL only has to discover when speaking pays and what to do about what is heard
for model, tag in (("models/team_bc", "laya_trained_grounded_speech"), ("models/team_bc_von", "von_trained")):
    run(["mp.pretrain_speech", "--model", model], "logs_pretrain_speech_%s.txt" % os.path.basename(model),
        skip_if=model + "/speech_report.json")
    run(["mp.evaluate", "--team", "trained", "--model", model, "--games", "20", "--tag", tag], "logs_eval_%s.txt" % tag,
        skip_if="results/%s.json" % tag)
    save("Results: %s with a speech decoder pretrained to describe what it sees" % model)

run(["mp.train_rl", "--model", "models/team_bc", "--out", "models/team_rl", "--iters", iters, "--resume"], "logs_train_rl.txt",
    skip_if="models/team_rl/train_report.json")
run(["mp.evaluate", "--team", "trained", "--model", "models/team_rl", "--games", "20", "--tag", "laya_rl"], "logs_eval_laya_rl.txt",
    skip_if="results/laya_rl.json")
save("Results: Laya after reinforcement learning (team reward, speech on)")
# the comparison the project is about: the same RL, but nobody can speak or hear
run(["mp.train_rl", "--model", "models/team_bc", "--out", "models/team_rl_mute", "--iters", iters, "--no-speech", "--resume"],
    "logs_train_rl_mute.txt", skip_if="models/team_rl_mute/train_report.json")
run(["mp.evaluate", "--team", "trained", "--model", "models/team_rl_mute", "--games", "20", "--no-speech", "--tag", "laya_rl_mute"],
    "logs_eval_laya_rl_mute.txt", skip_if="results/laya_rl_mute.json")
save("Results: Laya after reinforcement learning with the speech channel off")
print("rl queue finished   [%s]" % time.strftime("%H:%M"), flush=True)
