# backfill_trades_to_csv.py
# Schrijft recente Bitvavo-trades terug in data/live_trades.csv
# De-dupe t.o.v. bestaande CSV-regels.

import os, csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ccxt

API_KEY  = os.getenv("API_KEY",""); API_SECRET = os.getenv("API_SECRET","")
EXCHANGE = os.getenv("EXCHANGE","bitvavo").strip().lower()
DATA_DIR = Path(os.getenv("DATA_DIR","data")); DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADES_CSV = DATA_DIR / "live_trades.csv"
COINS_CSV = os.getenv("COINS","BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
OPERATOR_ID = os.getenv("OPERATOR_ID","").strip()

pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]

def load_existing_keys():
    keys = set()
    if not TRADES_CSV.exists(): return keys
    with TRADES_CSV.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            k = (r.get("timestamp",""), r.get("pair",""), r.get("side",""), r.get("qty",""))
            keys.add(k)
    return keys

def ensure_header():
    if not TRADES_CSV.exists():
        with TRADES_CSV.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","pair","side","avg_price","qty","eur","cash_eur",
                        "base","base_qty","pnl_eur","comment"])

def main():
    if not API_KEY or not API_SECRET:
        raise SystemExit("API keys ontbreken")

    klass = getattr(ccxt, EXCHANGE)
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    ex.load_markets()

    since = int((datetime.now(timezone.utc) - timedelta(days=3)).timestamp() * 1000)  # laatste 3 dagen
    existing = load_existing_keys()
    ensure_header()

    with TRADES_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for pair in pairs:
            try:
                trs = ex.fetch_my_trades(pair, since=since, limit=1000)
            except Exception:
                trs = []
            # filter optioneel op operatorId als je alleen bot-trades wilt
            if OPERATOR_ID:
                tmp = []
                for t in trs:
                    if str((t.get("info") or {}).get("operatorId") or "") == OPERATOR_ID:
                        tmp.append(t)
                trs = tmp
            for t in trs:
                ts = t.get("datetime") or datetime.fromtimestamp((t.get("timestamp") or 0)/1000, tz=timezone.utc).isoformat()
                side = (t.get("side") or "").upper()
                px = float(t.get("price") or 0.0)
                amt = float(t.get("amount") or 0.0)
                eur = px * amt
                base = pair.split("/")[0]
                key = (ts, pair, side, f"{amt:.8f}")
                if key in existing:
                    continue
                w.writerow([ts, pair, side, f"{px:.6f}", f"{amt:.8f}", f"{eur:.2f}", "",
                            base, f"{amt:.8f}", "", "backfill"])
                existing.add(key)
    print("Backfill voltooid.")

if __name__ == "__main__":
    main()
