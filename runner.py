# runner.py — eenvoudige orchestrator voor paper grid + rapportages
# - Start paper_grid.py en herstart bij crash
# - Draait report_paper.py periodiek (default: elke 6 uur)
# - Heartbeat met laatste equity uit /var/data/equity.csv

import os
import time
import sys
import csv
import subprocess
from pathlib import Path

# ====== Instellingen via ENV ======
REPORT_EVERY_HOURS   = float(os.getenv("REPORT_EVERY_HOURS", "6"))   # rapport elke 6 uur
SLEEP_HEARTBEAT_SEC  = int(os.getenv("SLEEP_HEARTBEAT_SEC", "300"))  # heartbeat elke 5 min
DATA_DIR             = Path(os.getenv("DATA_DIR", "/var/data"))      # opslagmap state/logs

# Bestanden
EQUITY_CSV = DATA_DIR / "equity.csv"

def latest_equity() -> str:
    """Lees laatste equity uit equity.csv (2e kolom)."""
    try:
        if not EQUITY_CSV.exists():
            return "n/a"
        # laatste niet-lege regel vinden
        last_row = None
        with EQUITY_CSV.open("r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if row and len(row) >= 2:
                    last_row = row
        if last_row is None:
            return "n/a"
        return f"€{float(last_row[1]):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "n/a"

def start_grid() -> subprocess.Popen:
    """Start paper_grid.py met on-buffered output."""
    print("[runner] start paper_grid.py …", flush=True)
    # -u voor realtime logs
    return subprocess.Popen([sys.executable, "-u", "paper_grid.py"])

def run_report():
    """Run report_paper.py éénmalig."""
    try:
        print("[runner] maak rapport …", flush=True)
        subprocess.run([sys.executable, "-u", "report_paper.py"], check=True)
        print(f"[runner] rapport klaar. Wacht {int(REPORT_EVERY_HOURS)} uur …", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[runner] rapport mislukt: {e}", flush=True)

def main():
    # Zorg dat DATA_DIR bestaat
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Start grid-proces
    grid_proc = start_grid()

    # Forceer meteen een eerste rapport bij start
    last_report_ts = 0.0
    run_report()
    last_report_ts = time.time()

    # Hoofdloop: bewaken + timers
    while True:
        # Herstart grid als het gestopt is
        if grid_proc.poll() is not None:
            print("[runner] grid gestopt; opnieuw starten …", flush=True)
            time.sleep(2)
            grid_proc = start_grid()

        # Heartbeat (laatste equity uit csv)
        eq = latest_equity()
        print(f"[runner] Heartbeat → Equity: {eq}", flush=True)

        # Tijd voor volgende rapportage?
        if (time.time() - last_report_ts) >= REPORT_EVERY_HOURS * 3600:
            run_report()
            last_report_ts = time.time()

        time.sleep(SLEEP_HEARTBEAT_SEC)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Gestopt.", flush=True)
