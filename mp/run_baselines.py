"""Run the non-learning team baselines back to back. python -m mp.run_baselines [games]"""
import subprocess
import sys

games = sys.argv[1] if len(sys.argv) > 1 else "50"
for extra in (["--team", "random"], ["--team", "scripted"], ["--team", "scripted_silent"],
              ["--team", "scripted", "--mute", "--tag", "scripted_muted"]):
    subprocess.run([sys.executable, "-m", "mp.evaluate", "--games", games] + extra, check=False)
