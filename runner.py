# runner.py — simpele orchestrator
# Start paper_grid.py en draai elke 12 uur report_paper.py
import subprocess, time, os, sys

REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "12"))
SLEEP_HEARTBEAT_SEC = 300  # kleine heartbeat, geen extra logging

def start_grid():
    # -u voor onbuffered logs
    return subprocess.Popen([sys.executable, "-u", "paper_grid.py"])

def run_report():
    print("[runner] maak rapport …")
    try:
        subprocess.run([sys.executable, "-u", "report_paper.py"], check=True)
        print("[runner] rapport klaar. Wacht", int(REPORT_EVERY_HOURS), "uur …")
    except Exception as e:
        print("[runner] rapport mislukt:", e)

def main():
    print(f"[runner] start paper_grid.py …")
    grid_proc = start_grid()

    # eerste rapport direct bij start
    last_report_ts = 0.0
    run_report()
    last_report_ts = time.time()

    # hoofdloop: alleen timers bijhouden en proces bewaken
    while True:
        # als grid-proces crasht, opnieuw starten
        if grid_proc.poll() is not None:
            print("[runner] grid gestopt; opnieuw starten …")
            grid_proc = start_grid()

        # elke REPORT_EVERY_HOURS opnieuw rapport maken
        if time.time() - last_report_ts >= REPORT_EVERY_HOURS * 3600:
            run_report()
            last_report_ts = time.time()

        time.sleep(SLEEP_HEARTBEAT_SEC)

if __name__ == "__main__":
    main()
