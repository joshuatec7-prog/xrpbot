# runner.py
# Start de paper grid bot en maak elke N uur een rapport
# - Default: 5 coins (BTC, ETH, SOL, XRP, LTC) met gewichten 35/30/15/10/10
# - Leest ENV COINS/WEIGHTS/DATA_DIR als je die opgeeft; anders gebruikt defaults
# - Start paper_grid.py als child process en draait rapportage elke 12 uur

import os
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ========= Instellingen (kun je desgewenst via ENV overriden) =========
DEFAULT_COINS = ['BTC/EUR', 'ETH/EUR', 'SOL/EUR', 'XRP/EUR', 'LTC/EUR']
DEFAULT_WEIGHTS = {
    'BTC/EUR': 0.35,
    'ETH/EUR': 0.30,
    'SOL/EUR': 0.15,
    'XRP/EUR': 0.10,
    'LTC/EUR': 0.10,
}
# Waar trades.csv/equity.csv geschreven/gelezen worden (rapport)
DATA_DIR = os.getenv("DATA_DIR", "/var/data")
# Hoe vaak rapport maken (uren)
REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "12"))

# ========= Helpers =========
def parse_coins_from_env() -> list:
    csv = os.getenv("COINS", "").strip()
    if not csv:
        return DEFAULT_COINS
    return [c.strip().upper() for c in csv.split(",") if c.strip()]

def parse_weights_from_env(coins: list) -> dict:
    txt = os.getenv("WEIGHTS", "").strip()
    if not txt:
        return DEFAULT_WEIGHTS.copy()
    out = {}
    for part in txt.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        k = k.strip().upper()
        try:
            out[k] = float(v)
        except Exception:
            pass
    # Alleen gewichten voor coins die we gebruiken
    out = {k: out.get(k, 0.0) for k in coins}
    s = sum(out.values())
    if s <= 0:
        # fallback naar defaults
        return {k: DEFAULT_WEIGHTS.get(k, 0.0) for k in coins}
    # normaliseren (samen = 1)
    return {k: (out[k] / s) for k in coins}

def ensure_dirs():
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    # subdir voor rapporten is optioneel, maar handig
    (Path(DATA_DIR) / "reports").mkdir(parents=True, exist_ok=True)

def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ========= Main runner =========
def main():
    ensure_dirs()

    coins = parse_coins_from_env()
    weights = parse_weights_from_env(coins)

    # Zet COINS/WEIGHTS expliciet in os.environ voor paper_grid.py
    os.environ["COINS"] = ",".join(coins)
    os.environ["WEIGHTS"] = ",".join(f"{k}:{weights[k]:.12f}" for k in coins)
    os.environ["DATA_DIR"] = DATA_DIR  # voor report_paper.py

    # Log setup
    alloc_str = ", ".join(f"{k}:{weights[k]:.2f}" for k in coins)
    print(f"[runner] {ts()}  DATA_DIR={DATA_DIR} | rapport iedere {REPORT_EVERY_HOURS:g} uur")
    print(f"[runner] {ts()}  start paper_grid.py …")
    print(f"== PAPER GRID Scenario C | capital via ENV | coins={coins} | alloc={{ {alloc_str} }} ==")

    # Start paper grid als child process (herstart bij exit)
    grid_proc = None
    last_report = 0.0

    while True:
        # 1) Zorg dat paper_grid.py loopt
        if grid_proc is None or grid_proc.poll() is not None:
            try:
                # -u voor line-buffered logs
                grid_proc = subprocess.Popen(
                    ["python", "-u", "paper_grid.py"],
                    stdout=None, stderr=None
                )
                print(f"[runner] {ts()}  paper_grid.py gestart (pid={grid_proc.pid})")
            except Exception as e:
                print(f"[runner] {ts()}  FOUT bij starten paper_grid.py: {e}")
                time.sleep(5)
                continue

        # 2) Maak rapport als interval voorbij is
        now = time.time()
        if (now - last_report) >= REPORT_EVERY_HOURS * 3600:
            try:
                print(f"[runner] {ts()}  maak rapport …")
                # éénmalig rapport; report_paper.py print in logs
                subprocess.run(["python", "-u", "report_paper.py"], check=False)
                print(f"[runner] {ts()}  rapport klaar. Wacht {REPORT_EVERY_HOURS:g} uur …")
            except Exception as e:
                print(f"[runner] {ts()}  FOUT bij rapport: {e}")
            last_report = time.time()

        # 3) Even slapen en dan weer checken
        time.sleep(10)

if __name__ == "__main__":
    main()
