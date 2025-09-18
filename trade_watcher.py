# trade_watcher.py — log alle niet-bot Bitvavo trades naar data/live_trades.csv
# - Backfill laatste N dagen
# - Pollt periodiek
# - Dedupe met seen_trades.json op trade-id
import os, time, json, csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
import ccxt

API_KEY  = os.getenv("API_KEY",""); API_SECRET = os.getenv("API_SECRET","")
EXCHANGE = os.getenv("EXCHANGE","bitvavo").strip().lower()
DATA_DIR = Path(os.getenv("DATA_DIR","data"))
COINS_CSV = os.getenv("COINS","BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
OPERATOR_ID = os.getenv("OPERATOR_ID","").strip()  # bot-operator; wordt uitgesloten
TRADES_BACKFILL_DAYS = int(os.getenv("TRADES_BACKFILL_DAYS","7"))
TRADES_POLL_SEC = int(os.getenv("TRADES_POLL_SEC","20"))

TRADES_CSV = DATA_DIR / "live_trades.csv"
SEEN_JSON  = DATA_DIR / "seen_trades.json"

GREEN="\033[92m"; RED="\033[91m"; CYAN="\033[96m"; RESET="\033[0m"

def now_iso():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def ensure_paths():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not TRADES_CSV.exists():
        with TRADES_CSV.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["timestamp","pair","side","avg_price","qty","eur","cash_eur",
                 "base","base_qty","pnl_eur","comment"]
            )

def load_seen():
    try:
        return set(json.loads(SEEN_JSON.read_text(encoding="utf-8")))
    except Exception:
        return set()

def save_seen(s):
    tmp = SEEN_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(s)), encoding="utf-8")
    tmp.replace(SEEN_JSON)

def main():
    if not API_KEY or not API_SECRET:
        raise SystemExit("API keys ontbreken voor trade_watcher.")
    ensure_paths()
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]

    klass = getattr(ccxt, EXCHANGE)
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    ex.load_markets()

    since_ms = int((datetime.now(timezone.utc) - timedelta(days=TRADES_BACKFILL_DAYS)).timestamp() * 1000)
    seen = load_seen()

    print(f"{CYAN}[watcher] start | backfill={TRADES_BACKFILL_DAYS}d | poll={TRADES_POLL_SEC}s{RESET}")

    while True:
        try:
            wrote = 0
            with TRADES_CSV.open("a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for pair in pairs:
                    try:
                        trades = ex.fetch_my_trades(pair, since=since_ms, limit=1000) or []
                    except Exception:
                        trades = []
                    for t in trades:
                        tid = str(t.get("id") or f"{pair}-{t.get('timestamp')}-{t.get('amount')}")
                        if tid in seen:
                            continue
                        opid = str((t.get("info") or {}).get("operatorId") or "")
                        if OPERATOR_ID and opid == OPERATOR_ID:
                            # bot-trade → bot logt zelf al; alleen markeren als gezien
                            seen.add(tid)
                            continue
                        ts = t.get("datetime") or datetime.fromtimestamp((t.get("timestamp") or 0)/1000, tz=timezone.utc).isoformat()
                        side = (t.get("side") or "").upper()
                        px   = float(t.get("price") or 0.0)
                        amt  = float(t.get("amount") or 0.0)
                        eur  = px * amt
                        base = pair.split("/")[0]
                        w.writerow([ts, pair, side, f"{px:.6f}", f"{amt:.8f}", f"{eur:.2f}", "",
                                    base, f"{amt:.8f}", "", "exchange"])
                        seen.add(tid); wrote += 1
                        col = GREEN if side=="SELL" else CYAN
                        print(f"{col}[watcher] {side} {pair} €{px:.6f} | amt={amt:.8f} | eur=€{eur:.2f}{RESET}")
            if wrote:
                save_seen(seen)
            time.sleep(TRADES_POLL_SEC)
        except KeyboardInterrupt:
            print("watcher gestopt."); break
        except Exception as e:
            print(f"[watcher] {e}"); time.sleep(5)

if __name__ == "__main__":
    main()
