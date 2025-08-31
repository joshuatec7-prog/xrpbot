# runner.py
# Start de paper grid bot + maak elke 12 uur een rapport
import os
import time
import threading
import subprocess
from pathlib import Path

# ---- Instellingen ----
# Waar de bot z’n data (trades.csv / equity.csv / reports/) schrijft
DATA_DIR = os.getenv("DATA_DIR", "/var/data")
REPORT_INTERVAL_HOURS = int(os.getenv("REPORT_INTERVAL_HOURS", "12"))  # elke 12 uur

# Bestanden in je repo
PAPER_GRID_FILE = "paper_grid.py"
REPORT_FILE     = "report_paper.py"

def ensure_dirs():
    """Zorg dat /var/data en /var/data/reports bestaan."""
    base = Path(DATA_DIR)
    reports = base / "reports"
    base.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

def run_paper_grid():
    """Draai paper_grid.py voor altijd (herstart bij crash)."""
    while True:
        try:
            print("[runner] start paper_grid.py …")
            proc = subprocess.Popen(
                ["python", "-u", PAPER_GRID_FILE],
                env={**os.environ, "DATA_DIR": DATA_DIR},
            )
            # blokkeer tot het proces eindigt
            proc.wait()
            code = proc.returncode
            print(f"[runner] paper_grid.py gestopt (exit={code}). Herstart in 5s …")
            time.sleep(5)
        except Exception as e:
            print(f"[runner] fout in run_paper_grid: {e}. Retry in 5s …")
            time.sleep(5)

def run_reports_forever():
    """Maak elke REPORT_INTERVAL_HOURS een rapport via report_paper.py."""
    # kleine delay zodat paper_grid al draait
    time.sleep(10)
    while True:
        try:
            print("[runner] maak rapport …")
            # report_paper.py leest DATA_DIR en schrijft naar /reports
            out = subprocess.run(
                ["python", "-u", REPORT_FILE],
                env={**os.environ, "DATA_DIR": DATA_DIR},
                capture_output=True,
                text=True,
            )
            if out.stdout:
                print(out.stdout.strip())
            if out.stderr:
                print(out.stderr.strip())
            print(f"[runner] rapport klaar. Wacht {REPORT_INTERVAL_HOURS} uur …")
        except Exception as e:
            print(f"[runner] fout in run_reports_forever: {e}")
        # altijd wachten, ook bij fout, zodat de loop doorloopt
        time.sleep(REPORT_INTERVAL_HOURS * 3600)

if __name__ == "__main__":
    print(f"[runner] DATA_DIR={DATA_DIR} | rapport iedere {REPORT_INTERVAL_HOURS} uur")
    ensure_dirs()

    # Start paper grid in een thread
    t = threading.Thread(target=run_paper_grid, daemon=True)
    t.start()

    # In de main-thread blijven we rapporten maken
    run_reports_forever()
