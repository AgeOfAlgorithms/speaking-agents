"""Zero-shot text decision models as full teams. python -m mp.run_zeroshot [games]"""
import subprocess
import sys

games = sys.argv[1] if len(sys.argv) > 1 else "20"
for extra in (["--team", "laya", "--tag", "laya_zeroshot"], ["--team", "laya", "--argmax", "--tag", "laya_zeroshot_greedy"],
              ["--team", "von", "--tag", "von_zeroshot"], ["--team", "von", "--argmax", "--tag", "von_zeroshot_greedy"]):
    subprocess.run([sys.executable, "-m", "mp.evaluate", "--games", games] + extra, check=False)
