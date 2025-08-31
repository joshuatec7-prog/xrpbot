# runner.py
import subprocess, threading, time, os, sys
from datetime import datetime

HOURS_BETWEEN_REPORTS = float(os.getenv("REPORT_EVERY_HOURS", "12"))

def log(msg):
    print(f"[RUNNER {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def start_paper_bot():
    """Start de paper grid bot als child-proces en herstart bij crash."""
    while True:
        log("Start paper_grid.py")
        p = subprocess.Popen([sys.executable, "paper_grid.py"])
        code = p.wait()
        log(f"paper_grid.py exited met code={code}. Herstart in 5s...")
        time.sleep(5)

def report_loop():
    """Draait elke X uur het rapport-script."""
    while True:
        try:
            log("Run report_paper.py")
            subprocess.run([sys.executable, "python_report_paper.py"], check=False)
            log("Rapport klaar")
        except Exception as e:
            log(f"Fout bij rapport: {e}")
        # wacht X uur
        sleep_sec = int(HOURS_BETWEEN_REPORTS * 3600)
        log(f"Wacht {sleep_sec//3600} uur tot volgend rapport...")
        time.sleep(sleep_sec)

if __name__ == "__main__":
    # Thread 1: bot
    t1 = threading.Thread(target=start_paper_bot, daemon=True)
    t1.start()

    # Thread 2: rapportage-loop
    t2 = threading.Thread(target=report_loop, daemon=True)
    t2.start()

    # Houd de runner in leven
    t1.join()
