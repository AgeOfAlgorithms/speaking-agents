"""One-off for the first warm-started model (models/team_bc), which was trained before mp.tune_speech
existed: once the pipeline has finished with it, tune its speech decoder and evaluate it again. Models
trained later get the speech phase from mp.train_bc itself and do not need this.

    python -m mp.run_speech_fix
"""
import os
import subprocess
import sys
import time

PY = [sys.executable, "-u", "-m"]
GIT = ["git", "-c", "user.name=seannam", "-c", "user.email=seannam@protonmail.com"]

while "pipeline finished" not in open("logs_pipeline.txt", encoding="utf-8").read():
    time.sleep(30)
if os.path.exists("results/laya_trained.json") and not os.path.exists("results/laya_trained_before_speech_tuning.json"):
    os.replace("results/laya_trained.json", "results/laya_trained_before_speech_tuning.json")
for args, log in ((["mp.tune_speech", "--model", "models/team_bc"], "logs_tune_speech.txt"),
                  (["mp.evaluate", "--team", "trained", "--model", "models/team_bc", "--games", "20", "--tag", "laya_trained"], "logs_eval_trained.txt")):
    print(">> %s   [%s]" % (" ".join(args), time.strftime("%H:%M")), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        subprocess.run(PY + args, stdout=f, stderr=subprocess.STDOUT, check=False)
subprocess.run(["git", "add", "-A"], check=False)
body = "Results: warm-started Laya after the speech-tuning phase\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
if subprocess.run(GIT + ["commit", "-q", "-m", body], check=False).returncode == 0:
    subprocess.run(["git", "push", "-q", "origin", "main"], env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), check=False)
print("speech fix finished   [%s]" % time.strftime("%H:%M"), flush=True)
