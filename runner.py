# runner.py — start live_grid.py en maak elke REPORT_EVERY_HOURS een rapport
import os, time, sys, subprocess, csv
from pathlib import Path

REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "6"))
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC", "300"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))  # zelfde default als live_grid.py
EQUITY_CSV = DATA_DIR / "live_equity.csv"       # zelfde bestandsnaam als live_grid.py

def latest_equity() -> str:
    try:
        if not EQUITY_CSV.exists():
            return "n/a"
        last = None
        with EQUITY_CSV.open("r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if row and len(row) >= 2:
                    last = row
        if last is None:
            return "n/a"
        return f"{float(last[1]):.2f}".replace(".", ",")
    except Exception:
        return "n/a"

def start_grid():
    print("[runner] start live_grid.py …", flush=True)
    return subprocess.Popen([sys.executable, "-u", "live_grid.py"])

def run_report_once():
    try:
        print("[runner] rapport maken …", flush=True)
        subprocess.run([sys.executable, "-u", "report_live.py"], check=True)
        print(f"[runner] rapport klaar. Wacht {int(REPORT_EVERY_HOURS)} uur …", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[runner] rapport mislukt: {e}", flush=True)

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    grid = start_grid()

    # meteen eerste rapport
    last_report_ts = 0.0
    run_report_once()
    last_report_ts = time.time()

    try:
        while True:
            # herstart grid indien nodig
            if grid.poll() is not None:
                print("[runner] grid gestopt; opnieuw starten …", flush=True)
                time.sleep(2)
                grid = start_grid()

            # heartbeat
            eq = latest_equity()
            print(f"[runner] Heartbeat → Equity: €{eq}", flush=True)

            # rapport timer
            if (time.time() - last_report_ts) >= REPORT_EVERY_HOURS * 3600:
                run_report_once()
                last_report_ts = time.time()

            time.sleep(SLEEP_HEARTBEAT_SEC)
    except KeyboardInterrupt:
        print("Gestopt.", flush=True)

if __name__ == "__main__":
    main()
