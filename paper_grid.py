# paper_grid.py — Scenario C (max rendement, agressief)
# - Paper grid trading simulator
# - 2 coins (ETH + SOL), alloc 60/40
# - Grote tickets (ORDER_SIZE_FACTOR=2.0)
# - 12 grid-levels, smalle band ±15%
# - Persistente state + trades.csv + equity.csv

import os, json, time, random, csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Dict

import ccxt
import pandas as pd

# ------------------ Defaults Scenario C ------------------
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "15000"))

# Alleen ETH en SOL (hoger rendement, meer beweging)
COINS = ["ETH/EUR", "SOL/EUR"]

# Allocatie 60/40
WEIGHTS = {"ETH/EUR": 0.6, "SOL/EUR": 0.4}

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "12"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.15"))     # ±15% band
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))    # 0.15% per order
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "30"))
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR", "2.0"))

MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
MIN_TRADE_EUR       = float(os.getenv("MIN_TRADE_EUR", "5"))

EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")
DATA_DIR    = Path(os.getenv("DATA_DIR", "."))

STATE_FILE  = DATA_DIR / "state.json"
TRADES_CSV  = DATA_DIR / "trades.csv"
EQUITY_CSV  = DATA_DIR / "equity.csv"

# ------------------ Helpers ------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

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
                w.writerow(["timestamp","pair","side","price","amount","fee_eur","eur_cash","coin","coin_qty","realized_pnl_eur","comment"])
            elif path == EQUITY_CSV:
                w.writerow(["date","total_equity_eur"])
        w.writerow(row)

# ------------------ Exchange ------------------
def make_ex():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"enableRateLimit": True})
    ex.load_markets()
    return ex

# ------------------ Grid bouw ------------------
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    """30d 1h closes → P10–P90. Fallback: ±15% band rond laatste prijs."""
    try:
        ohlcv = ex.fetch_ohlcv(pair, timeframe="1h", limit=24*30)
        if ohlcv and len(ohlcv) >= 50:
            closes = [c[4] for c in ohlcv]
            s = pd.Series(closes)
            p10 = float(s.quantile(0.10))
            p90 = float(s.quantile(0.90))
            if p90 > p10 > 0:
                return p10, p90
    except Exception:
        pass
    last = float(ex.fetch_ticker(pair)["last"])
    return last * (1 - BAND_PCT), last * (1 + BAND_PCT)

# ------------------ Portfolio ------------------
def init_portfolio() -> dict:
    port = {"EUR": CAPITAL_EUR, "coins": {}, "pnl_realized": 0.0}
    for p in COINS:
        port["coins"][p] = {"qty": 0.0, "cash_alloc": CAPITAL_EUR * WEIGHTS[p]}
    return port

def mark_to_market(ex, state: dict) -> float:
    total = state["portfolio"]["EUR"]
    for p in COINS:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

# ------------------ Grid state ------------------
def mk_grid_state(ex, pair: str) -> dict:
    low, high = compute_band_from_history(ex, pair)
    grid_levels = geometric_levels(low, high, GRID_LEVELS)
    return {
        "low": low,
        "high": high,
        "levels": grid_levels,
        "last_price": None,
        "inventory_lots": [],
    }

# ------------------ Fill-simulator ------------------
def try_fill_grid(pair: str, price_now: float, price_prev: float, grid: dict, port: dict, ex) -> List[str]:
    logs = []
    levels = grid["levels"]

    # Cash allocatie per coin
    cash_alloc = port["coins"][pair]["cash_alloc"]
    base_order = (cash_alloc * 0.90) / max(1, (len(levels)//2))
    order_eur = max(MIN_TRADE_EUR, base_order * ORDER_SIZE_FACTOR)

    # BUY
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            if port["EUR"] <= MIN_CASH_BUFFER_EUR or order_eur > (port["EUR"] - MIN_CASH_BUFFER_EUR):
                continue
            qty = order_eur / L
            fee_eur = order_eur * FEE_PCT
            port["EUR"] -= (order_eur + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": L})
            append_csv(TRADES_CSV, [now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}", f"{port['EUR']:.2f}", pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", f"{0.0:.2f}", "grid_buy"])
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{L:.2f} | EUR_cash={port['EUR']:.2f}")

    # SELL
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
            port["EUR"] += (proceeds - fee_eur)
            port["coins"][pair]["qty"] -= qty
            port["pnl_realized"] += pnl
            append_csv(TRADES_CSV, [now_iso(), pair, "SELL", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}", f"{port['EUR']:.2f}", pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "grid_sell"])
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{L:.2f} | PnL={pnl:.2f} | EUR_cash={port['EUR']:.2f}")

    grid["last_price"] = price_now
    return logs

# ------------------ Main ------------------
def main():
    print(f"== PAPER GRID Scenario C | capital=€{CAPITAL_EUR:.2f} | coins={COINS} | alloc={WEIGHTS} | factor={ORDER_SIZE_FACTOR} ==")

    ex = make_ex()

    state = load_state() or {}
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio()

    if "grids" not in state:
        state["grids"] = {}
    for p in COINS:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p)

    save_state(state)
    last_equity_day = None

    while True:
        try:
            today = datetime.now(timezone.utc).date().isoformat()
            if today != last_equity_day:
                equity = mark_to_market(ex, state)
                append_csv(EQUITY_CSV, [today, f"{equity:.2f}"])
                last_equity_day = today

            for p in COINS:
                t = ex.fetch_ticker(p)
                px = float(t["last"])
                grid = state["grids"][p]
                logs = try_fill_grid(p, px, grid["last_price"], grid, state["portfolio"], ex)
                if logs:
                    print("\n".join(logs))

            save_state(state)
            time.sleep(SLEEP_SEC)

        except Exception as e:
            print(f"[runtime] {e}")
            time.sleep(3)

if __name__ == "__main__":
    main()

