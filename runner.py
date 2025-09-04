# runner.py – eenvoudige orchestrator voor paper_grid.py + periodieke report
# - Start paper_grid.py en bewaakt het proces (restart als het zou stoppen)
# - Draait report_paper.py elke REPORT_EVERY_HOURS (default 6)
# - Heartbeat (SLEEP_HEARTBEAT_SEC) logt laatste equity
# - RESET_STATE & WARM_START:
#     * RESET_STATE=true  -> verwijder state.json (verse grid)
#     * WARM_START=true   -> laat trades.csv & equity.csv staan
#     * CLEAR_HISTORY=true -> wis ook trades.csv & equity.csv
#
# Start command op Render:  python -u runner.py

import os
import sys
import time
import csv
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ---------- ENV ----------
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "6"))
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC", "300"))  # 5 min
RESET_STATE = os.getenv("RESET_STATE", "false").lower() in ("1", "true", "yes")
WARM_START = os.getenv("WARM_START", "true").lower() in ("1", "true", "yes")
CLEAR_HISTORY = os.getenv("CLEAR_HISTORY", "false").lower() in ("1", "true", "yes")

STATE_FILE = DATA_DIR / "state.json"
TRADES_CSV = DATA_DIR / "trades.csv"
EQUITY_CSV = DATA_DIR / "equity.csv"

# ---------- helpers ----------
def ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def prepare_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RESET_STATE:
        # verse grid: verwijder alleen state.json
        if STATE_FILE.exists():
            STATE_FILE.unlink(missing_ok=True)
        # optioneel: historie weggooien
        if CLEAR_HISTORY:
            if TRADES_CSV.exists():
                TRADES_CSV.unlink(missing_ok=True)
            if EQUITY_CSV.exists():
                EQUITY_CSV.unlink(missing_ok=True)
        print(f"[runner] RESET_STATE actief | warm_start={WARM_START} | clear_history={CLEAR_HISTORY}")

def start_grid() -> subprocess.Popen:
    # -u: unbuffered output voor live logs
    print("[runner] start paper_grid.py …")
    return subprocess.Popen([sys.executable, "-u", "paper_grid.py"])

def run_report_once():
    try:
        print("[runner] maak rapport …")
        subprocess.run([sys.executable, "-u", "report_paper.py"], check=True)
        print(f"[runner] rapport klaar. Wacht {int(REPORT_EVERY_HOURS)} uur …")
    except Exception as e:
        print(f"[runner] rapport mislukt: {e}")

def read_last_equity() -> float | None:
    if not EQUITY_CSV.exists():
        return None
    try:
        with EQUITY_CSV.open("r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) >= 2:
            # laatste regel: [date, total_equity_eur]
            return float(rows[-1][1])
    except Exception:
        return None
    return None

def watchdog(proc: subprocess.Popen) -> subprocess.Popen:
    # Als paper_grid.py omvalt, opnieuw starten
    if proc.poll() is not None:
        print("[runner] paper_grid.py gestopt; opnieuw starten …")
        return start_grid()
    return proc

# ---------- main ----------
def main():
    prepare_data_dir()

    grid_proc = start_grid()

    # Eerste rapport direct bij start (handig voor controle)
    last_report_ts = 0.0
    run_report_once()
    last_report_ts = time.time()

    # Hoofdloop: heartbeat + timers
    last_heartbeat = 0.0
    report_interval_sec = REPORT_EVERY_HOURS * 3600.0

    while True:
        # grid bewaken
        grid_proc = watchdog(grid_proc)

        now = time.time()

        # Heartbeat
        if now - last_heartbeat >= SLEEP_HEARTBEAT_SEC:
            eq = read_last_equity()
            if eq is not None:
                print(f"[{ts()}] [runner] Heartbeat → Equity: €{eq:,.2f}")
            else:
                print(f"[{ts()}] [runner] Heartbeat … (nog geen equity.csv)")
            last_heartbeat = now

        # Periodiek rapport
        if now - last_report_ts >= report_interval_sec:
            run_report_once()
            last_report_ts = now

        # kleine slaap
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[runner] Stop.")
