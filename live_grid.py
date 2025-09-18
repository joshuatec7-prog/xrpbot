# adopt_bot_positions.py
# Bouw data/live_state.json opnieuw uit je Bitvavo-trades.
# - Filtert op OPERATOR_ID als die is gezet
# - FIFO-matcht buys vs sells → resterende open lots
# - baseline = free_base - bot_qty (beschermt je eigen startcoins)
# Run één keer: python3 adopt_bot_positions.py

import os, json
from pathlib import Path
from datetime import datetime, timezone
import ccxt
import pandas as pd

# ==== ENV ====
API_KEY   = os.getenv("API_KEY","")
API_SECRET= os.getenv("API_SECRET","")
EXCHANGE  = os.getenv("EXCHANGE","bitvavo").strip().lower()
DATA_DIR  = Path(os.getenv("DATA_DIR","data"))
COINS_CSV = os.getenv("COINS","BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
GRID_LEVELS = int(os.getenv("GRID_LEVELS","24"))
BAND_PCT  = float(os.getenv("BAND_PCT","0.20"))
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR","1000"))
WEIGHTS_CSV = os.getenv("WEIGHTS","BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()
OPERATOR_ID = os.getenv("OPERATOR_ID","").strip()

STATE_FILE = DATA_DIR / "live_state.json"

def normalize_weights(pairs, weights_csv):
    d = {}
    for item in [x.strip() for x in weights_csv.split(",") if x.strip()]:
        if ":" in item:
            k, v = item.split(":",1)
            try: d[k.strip().upper()] = float(v)
            except: pass
    s = sum(d.values())
    if s <= 0: return {p: 1.0/len(pairs) for p in pairs}
    return {p: d.get(p,0.0)/s for p in pairs}

def geometric_levels(low, high, n):
    if low <= 0 or high <= 0 or n < 2: return [low, high]
    r = (high/low) ** (1/(n-1))
    return [low*(r**i) for i in range(n)]

def compute_band_from_history(ex, pair):
    try:
        ohlcv = ex.fetch_ohlcv(pair, timeframe="1h", limit=24*30)
        if ohlcv and len(ohlcv) >= 50:
            closes = [c[4] for c in ohlcv if c and c[4] is not None]
            s = pd.Series(closes)
            p10 = float(s.quantile(0.10)); p90 = float(s.quantile(0.90))
            if p90 > p10 > 0: return p10, p90
    except Exception:
        pass
    last = float(ex.fetch_ticker(pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def mk_grid_state(ex, pair, levels):
    low, high = compute_band_from_history(ex, pair)
    return {
        "pair": pair,
        "low": low, "high": high,
        "levels": geometric_levels(low, high, levels),
        "last_price": None,
        "inventory_lots": [],   # [{qty, buy_price}]
    }

def fifo_from_trades(trades):
    # trades: list of dicts with keys side('buy'/'sell'), amount, price, timestamp asc
    lots = []
    for t in trades:
        side = (t.get("side") or "").lower()
        amt = float(t.get("amount") or 0.0)
        px  = float(t.get("price") or 0.0)
        if amt <= 0 or px <= 0: 
            continue
        if side == "buy":
            lots.append([amt, px])  # [qty, price]
        elif side == "sell":
            need = amt
            i = 0
            while need > 1e-12 and i < len(lots):
                take = min(need, lots[i][0])
                lots[i][0] -= take
                need -= take
                if lots[i][0] <= 1e-12:
                    lots.pop(i)
                else:
                    i += 1
    out = []
    for qty, price in lots:
        if qty > 1e-12:
            out.append({"qty": qty, "buy_price": price})
    return out

def main():
    if not API_KEY or not API_SECRET:
        raise SystemExit("API keys ontbreken")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    klass = getattr(ccxt, EXCHANGE)
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    ex.load_markets()

    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten in COINS")

    # huidige free-balances → voor baseline
    try:
        free_bal = ex.fetch_balance().get("free", {}) or {}
    except Exception:
        free_bal = {}

    grids = {}
    coins_qty = {}
    baselines = {}

    for pair in pairs:
        base = pair.split("/")[0]
        try:
            trs = ex.fetch_my_trades(pair, limit=1000)
        except Exception:
            trs = []

        # filter op operatorId als gezet
        if OPERATOR_ID:
            filtered = []
            for t in trs:
                opid = str((t.get("info") or {}).get("operatorId") or "")
                if opid == OPERATOR_ID:
                    filtered.append(t)
            trs = filtered

        trs.sort(key=lambda t: int(t.get("timestamp") or 0))
        open_lots = fifo_from_trades(trs)

        free_base = float(free_bal.get(base) or 0.0)
        bot_qty = sum(l["qty"] for l in open_lots)
        baseline = max(0.0, free_base - bot_qty)

        g = mk_grid_state(ex, pair, GRID_LEVELS)
        g["inventory_lots"] = open_lots
        grids[pair] = g
        coins_qty[pair] = {"qty": bot_qty, "cash_alloc": 0.0}
        baselines[base] = baseline

    weights = normalize_weights(pairs, WEIGHTS_CSV)
    for p in pairs:
        if p in coins_qty:
            coins_qty[p]["cash_alloc"] = CAPITAL_EUR * weights[p]

    state = {
        "portfolio": {
            "play_cash_eur": CAPITAL_EUR,
            "cash_eur": CAPITAL_EUR,
            "pnl_realized": 0.0,
            "coins": coins_qty,
        },
        "grids": grids,
        "baseline": baselines,
    }

    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print("live_state.json opgebouwd.")
    for p in pairs:
        lots = grids[p]["inventory_lots"]
        print(f"{p}: lots={len(lots)} | qty_totaal={sum(l['qty'] for l in lots):.8f} | baseline {p.split('/')[0]}={baselines[p.split('/')[0]]:.8f}")

if __name__ == "__main__":
    main()
