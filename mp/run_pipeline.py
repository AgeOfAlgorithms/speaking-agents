"""Everything from the scripted baselines to an evaluated warm-started Laya, in order. Safe to re-run:
training resumes from its last checkpoint and finished stages are skipped.

    python -m mp.run_pipeline
"""
import os
import subprocess
import sys

PY = [sys.executable, "-u", "-m"]


def run(args, log, skip_if=None):
    if skip_if and os.path.exists(skip_if):
        print("skip (already have %s): %s" % (skip_if, " ".join(args)), flush=True)
        return
    print(">>", " ".join(args), flush=True)
    with open(log, "a", encoding="utf-8") as f:
        subprocess.run(PY + args, stdout=f, stderr=subprocess.STDOUT, check=False)


run(["mp.run_baselines", "50"], "logs_baselines.txt", skip_if="results/scripted_muted.json")
run(["mp.collect", "--samples", "80000"], "logs_collect.txt", skip_if="data/warmstart/info.json")
run(["mp.train_bc", "--limit", "60000", "--out", "models/team_bc", "--resume"], "logs_train_team_bc.txt",
    skip_if="models/team_bc/train_report.json")
run(["mp.evaluate", "--team", "trained", "--model", "models/team_bc", "--games", "20", "--tag", "laya_trained"],
    "logs_eval_trained.txt", skip_if="results/laya_trained.json")
print("pipeline finished", flush=True)
