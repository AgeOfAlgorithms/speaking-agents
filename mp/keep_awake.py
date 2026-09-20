"""Stops Windows from idle-sleeping while training queues are running (a 30-minute idle timeout once put
the PC to sleep for 8 hours mid-run). Changes no system setting: the request lasts only while this process
lives, and it exits by itself once no `mp.` job is left.

    python -m mp.keep_awake
"""
import ctypes
import subprocess
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
JOBS = ("mp.run_", "mp.train_", "mp.evaluate", "mp.collect", "mp.tune_speech")


def jobs_running():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ForEach-Object CommandLine"],
                         capture_output=True, text=True).stdout
    return any(j in out for j in JOBS)


if sys.platform != "win32":
    sys.exit("only needed on Windows")
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print("keeping the PC awake while training jobs run", flush=True)
time.sleep(120)
while jobs_running():
    time.sleep(120)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print("no jobs left; sleep allowed again", flush=True)
