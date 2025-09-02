# runner.py — eenvoudige orkestrator voor de paper grid bot
# - Start paper_grid.py (blijft doorlopen; herstart bij crash)
# - Draait report_paper.py elke REPORT_EVERY_HOURS
# - Heartbeat elke SLEEP_HEARTBEAT_SEC met actuele equity

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

# ===== Instellingen =====
REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "6"))     # uren
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC", "300"))    # 5 min
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data")).resolve()

# ===== Helpers =====
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def start_grid():
    """Start de grid-bot in eigen proces (-u = onbuffered logs)."""
    print(f"[{ts()}] [runner] start paper_grid.py …")
    # gebruik sys.executable zodat dezelfde Python gebruikt wordt als op Render
    return subprocess.Popen([sys.executable, "-u", "paper_grid.py"])

def run_report():
    """Draai het rapport-script eenmalig."""
    try:
        print(f"[{ts()}] [runner] maak rapport …")
        subprocess.run([sys.executable, "-u", "report_paper.py"], check=True)
        print(f"[{ts()}] [runner] rapport klaar. Wacht {int(REPORT_EVERY_HOURS)} uur …")
    except subprocess.CalledProcessError as e:
        print(f"[{ts()}] [runner] rapport mislukt: exit {e.returncode}")
    except Exception as e:
        print(f"[{ts()}] [runner] rapport fout: {e}")

def heartbeat():
    """Log een mini-overzicht met de laatste equity uit DATA_DIR/equity.csv."""
    try:
        eq_file = DATA_DIR / "equity.csv"
        if not eq_file.exists():
            print(f"[{ts()}] [runtime] Heartbeat → geen equity.csv gevonden in {DATA_DIR}")
            return

        # pandas alleen binnen deze functie importeren
        import pandas as pd
        df = pd.read_csv(eq_file)
        if df.empty:
            print(f"[{ts()}] [runtime] Heartbeat → equity.csv is leeg")
            return

        last = df.iloc[-1]
        # sta toe dat kolomnaam iets anders is (total_equity_eur of equity_eur)
        equity_col = None
        for c in ("total_equity_eur", "equity_eur", "total_equity", "equity"):
            if c in df.columns:
                equity_col = c
                break

        if equity_col is None:
            # val terug op de laatste kolom
            equity_val = float(last.iloc[-1])
        else:
            equity_val = float(last[equity_col])

        print(f"[{ts()}] [runtime] Heartbeat → Equity: €{equity_val:,.2f}")
    except Exception as e:
        print(f"[{ts()}] [runtime] Heartbeat fout: {e}")

# ===== Hoofdprogramma =====
def main():
    # zorg dat data-dir bestaat (voor rapporten/CSV)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    grid_proc = start_grid()

    # eerste rapport direct bij start
    last_report_ts = 0.0
    run_report()
    last_report_ts = time.time()

    last_heartbeat_ts = 0.0

    while True:
        # Herstart grid als hij uitgevallen is
        if grid_proc.poll() is not None:
            print(f"[{ts()}] [runner] grid-proces gestopt (exit {grid_proc.returncode}); opnieuw starten …")
            time.sleep(2)
            grid_proc = start_grid()

        # Heartbeat
        now = time.time()
        if now - last_heartbeat_ts >= SLEEP_HEARTBEAT_SEC:
            heartbeat()
            last_heartbeat_ts = now

        # Rapport elke REPORT_EVERY_HOURS
        if now - last_report_ts >= REPORT_EVERY_HOURS * 3600:
            run_report()
            last_report_ts = time.time()

        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"[{ts()}] Gestopt met Ctrl+C")
