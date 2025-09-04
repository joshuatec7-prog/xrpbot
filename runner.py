# runner.py – start de grid-bot en maak elke X uur een rapport
import os, sys, time, subprocess

REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "6"))
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC", "300"))  # 5 min

def start_grid():
    # -u = onbuffered logging in Render
    return subprocess.Popen([sys.executable, "-u", "paper_grid.py"])

def run_report():
    try:
        subprocess.run([sys.executable, "-u", "report_paper.py"], check=True)
        print(f"[runner] rapport klaar. Wacht {int(REPORT_EVERY_HOURS)} uur …")
    except Exception as e:
        print(f"[runner] rapport mislukte: {e}")

def main():
    print(f"[runner] start paper_grid.py …")
    grid_proc = start_grid()

    # eerste rapport meteen bij start
    last_report_ts = 0.0
    run_report()
    last_report_ts = time.time()

    # orkestratie-lus
    while True:
        # als grid crasht → opnieuw starten
        if grid_proc.poll() is not None:
            print("[runner] grid-bot is gestopt; opnieuw starten …")
            grid_proc = start_grid()

        # periodiek rapport
        if time.time() - last_report_ts >= REPORT_EVERY_HOURS * 3600:
            run_report()
            last_report_ts = time.time()

        # klein heartbeat‐logje
        print("[runner] heartbeat …")
        time.sleep(SLEEP_HEARTBEAT_SEC)

if __name__ == "__main__":
    main()
