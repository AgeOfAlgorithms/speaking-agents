"""After everything else: the comparison that matters, on 50 held-out worlds instead of 20 (at 20 the team
score moves by a few points from luck alone, which is as large as the differences being looked for).

    python -m mp.run_final_eval
"""
import os
import subprocess
import sys
import time

PY = [sys.executable, "-u", "-m"]
GIT = ["git", "-c", "user.name=seannam", "-c", "user.email=seannam@protonmail.com"]

while "von queue finished" not in open("logs_von_queue.txt", encoding="utf-8").read():
    time.sleep(60)
for model, tag, extra in (("models/team_bc", "final50_before_rl", []),
                          ("models/team_bc", "final50_before_rl_mute", ["--no-speech"]),
                          ("models/team_rl_local", "final50_rl_speech", []),
                          ("models/team_rl_local_mute", "final50_rl_mute", ["--no-speech"])):
    if os.path.exists("results/%s.json" % tag) or not os.path.exists(model + "/model.safetensors"):
        continue
    print(">> evaluate %s as %s   [%s]" % (model, tag, time.strftime("%H:%M")), flush=True)
    with open("logs_eval_%s.txt" % tag, "a", encoding="utf-8") as f:
        subprocess.run(PY + ["mp.evaluate", "--team", "trained", "--model", model, "--games", "50", "--tag", tag] + extra,
                       stdout=f, stderr=subprocess.STDOUT, check=False)
subprocess.run(["git", "add", "-A"], check=False)
body = "Results: 50-world comparison, before RL / RL with speech / RL without\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
if subprocess.run(GIT + ["commit", "-q", "-m", body], check=False).returncode == 0:
    subprocess.run(["git", "push", "-q", "origin", "main"], env=dict(os.environ, GIT_TERMINAL_PROMPT="0"), check=False)
print("final eval finished   [%s]" % time.strftime("%H:%M"), flush=True)
