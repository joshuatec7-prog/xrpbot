# paper_grid.py
# 24/7 PAPER GRID BOT (Bitvavo public data only)
# - Grid per pair op basis van P10–P90 van 30d 1h data (fallback: median ± BAND_PCT)
# - Virtuele fills (paper), fees, PnL, persistente state
# - Logt trades.csv en equity.csv; state.json voor herstel

import os, json, time, math, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ------------------ Config via ENV ------------------
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "15000"))
COINS_CSV   = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR")
WEIGHTS_CSV = os.getenv("WEIGHTS", "").strip()

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))      # fallback ±20%
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))     # 0.15%
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "30"))
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")

# logopties
LOG_TRADES      = os.getenv("LOG_TRADES", "true").lower() in ("1","true","yes","y")
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC", "600"))

DATA_DIR   = Path(".")
STATE_FILE = DATA_DIR / "state_paper.json"
TRADES_CSV = DATA_DIR / "paper_trades.csv"
EQUITY_CSV = DATA_DIR / "paper_equity.csv"

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{now_iso()}] {msg}")

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

def append_csv(path: Path, row: List):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            if path == TRADES_CSV:
                w.writerow(["timestamp","pair","side","price","amount","fee_eur","cash_eur","coin","coin_qty","realized_pnl_eur","comment"])
            elif path == EQUITY_CSV:
                w.writerow(["date","total_equity_eur"])
        w.writerow(row)

# ------------------ Exchange (public) ------------------
def make_ex():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"enableRateLimit": True})
    ex.load_markets()
    return ex

def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    try:
        ohlcv = ex.fetch_ohlcv(pair, timeframe="1h", limit=24*30)
        if not ohlcv or len(ohlcv) < 50:
            raise ValueError("te weinig data")
        closes = [c[4] for c in ohlcv]
        s = pd.Series(closes)
        p10 = float(s.quantile(0.10))
        p90 = float(s.quantile(0.90))
        if p90 > p10 > 0:
            return p10, p90
    except Exception:
        pass
    last = float(ex.fetch_ticker(pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    if weights_csv:
        parts = [x.strip() for x in weights_csv.split(",") if x.strip()]
        d = {}
        for p in parts:
            k, v = p.split(":")
            d[k.strip().upper()] = float(v)
        s = sum(d.get(x, 0.0) for x in pairs)
        if s > 0:
            return {x: d.get(x, 0.0)/s for x in pairs}
    eq = 1.0 / len(pairs)
    return {x: eq for x in pairs}

def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    port = {"portfolio_eur": CAPITAL_EUR, "coins": {}, "pnl_realized": 0.0}
    for p in pairs:
        port["coins"][p] = {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]}
    return port

def mk_grid_state(ex, pair: str) -> dict:
    low, high = compute_band_from_history(ex, pair)
    return {
        "low": low, "high": high,
        "levels": geometric_levels(low, high, GRID_LEVELS),
        "last_price": None,
        "inventory_lots": [],  # [{qty, buy_price}]
    }

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    total = state["portfolio"]["portfolio_eur"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

def try_fill_grid(pair, price_now, price_prev, grid, port, ex) -> List[str]:
    logs = []
    levels = grid["levels"]
    cash_alloc = port["coins"][pair]["cash_alloc"]
    order_eur = max(1.0, (cash_alloc * 0.90) / (len(levels)//2))

    # BUY: down-cross
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            if port["portfolio_eur"] < order_eur*(1+FEE_PCT):
                logs.append(f"[{pair}] BUY skip: onvoldoende EUR")
                continue
            qty = order_eur / L
            fee_eur = order_eur * FEE_PCT
            port["portfolio_eur"] -= (order_eur + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": L})
            append_csv(TRADES_CSV, [now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                                    f"{port['portfolio_eur']:.2f}", pair.split("/")[0],
                                    f"{port['coins'][pair]['qty']:.8f}", "0.00", "grid_buy"])
            if LOG_TRADES:
                log(f"{pair} BUY  qty={qty:.8f}  px=€{L:.6f}  fee=€{fee_eur:.2f}  EUR_cash={port['portfolio_eur']:.2f}")

    # SELL: up-cross – pak eerste lot met winst
    if price_prev is not None and price_now > price_prev and grid["inventory_lots"]:
        crossed = [L for L in levels if price_prev < L <= price_now]
        for L in crossed:
            lot_idx = None
            for i, lot in enumerate(grid["inventory_lots"]):
                if L > lot["buy_price"]:
                    lot_idx = i
                    break
            if lot_idx is None:
                continue
            lot = grid["inventory_lots"].pop(lot_idx)
            qty = lot["qty"]
            proceeds = qty * L
            fee_eur = proceeds * FEE_PCT
            pnl = proceeds - fee_eur - (qty * lot["buy_price"])
            port["portfolio_eur"] += (proceeds - fee_eur)
            port["coins"][pair]["qty"] -= qty
            port["pnl_realized"] += pnl
            append_csv(TRADES_CSV, [now_iso(), pair, "SELL", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                                    f"{port['portfolio_eur']:.2f}", pair.split("/")[0],
                                    f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "grid_sell"])
            if LOG_TRADES:
                log(f"{pair} SELL qty={qty:.8f}  px=€{L:.6f}  pnl=€{pnl:.2f}  EUR_cash={port['portfolio_eur']:.2f}")

    grid["last_price"] = price_now
    return logs

def main():
    log(f"== PAPER GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% ==")
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    ex = make_ex()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden.")

    # weights + state
    weights = normalize_weights(pairs, WEIGHTS_CSV)
    state = load_state() or {}
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state: state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p)
    save_state(state)

    last_equity_day = None
    state["_last_summary_ts"] = 0.0

    while True:
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            if today != last_equity_day:
                equity = mark_to_market(ex, state, pairs)
                append_csv(EQUITY_CSV, [today, f"{equity:.2f}"])
                last_equity_day = today

            for p in pairs:
                px = float(ex.fetch_ticker(p)["last"])
                grid = state["grids"][p]
                try_fill_grid(p, px, grid["last_price"], grid, state["portfolio"], ex)

            if LOG_SUMMARY_SEC > 0:
                now_ts = time.time()
                if now_ts - state["_last_summary_ts"] >= LOG_SUMMARY_SEC:
                    eq = mark_to_market(ex, state, pairs)
                    eur = state["portfolio"]["portfolio_eur"]
                    realized = state["portfolio"]["pnl_realized"]
                    per = ", ".join(f"{p.split('/')[0]}:{state['portfolio']['coins'][p]['qty']:.6f}" for p in pairs)
                    log(f"SUMMARY equity=€{eq:.2f} EUR_cash=€{eur:.2f} realized=€{realized:.2f} pos[{per}]")
                    state["_last_summary_ts"] = now_ts

            save_state(state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            log(f"[neterr] {e}; backoff.."); time.sleep(2 + random.random())
        except Exception as e:
            log(f"[runtime] {e}"); time.sleep(5)

if __name__ == "__main__":
    main()

