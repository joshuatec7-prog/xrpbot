# paper_grid.py
# 24/7 PAPER GRID BOT (Bitvavo public data only)
# - Grid per pair op basis van P10–P90 van 30d 1h data (fallback: median ± BAND_PCT)
# - Virtuele fills (paper), fees, PnL, persistente state
# - Logt trades.csv en equity.csv; state.json voor herstel
# --------------------------------------------------------------

import os, json, time, math, csv, random, statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ------------------ Config via ENV ------------------
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "15000"))

# COINS als CSV: "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
COINS_CSV = os.getenv(
    "COINS",
    "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
)

# Weights als CSV "PAIR:WEIGHT,...", bv "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10"
WEIGHTS_CSV = os.getenv("WEIGHTS", "").strip()

# Grid instellingen
GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))     # fallback band ±20%
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))    # 0.15% per order
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "30"))      # loop interval
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")         # public data provider

DATA_DIR = Path(".")
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
                w.writerow(["timestamp","pair","side","price","amount","fee_eur","cash_eur","coin","coin_qty","realized_pnl_eur","comment"])
            elif path == EQUITY_CSV:
                w.writerow(["date","total_equity_eur"])
        w.writerow(row)

def pct(a, b):  # percentage change
    return (a - b) / b if b else 0.0

# ------------------ Exchange (public) ------------------
def make_ex():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"enableRateLimit": True})
    ex.load_markets()
    return ex

# ------------------ Grid bouw ------------------
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    # multiplicative spacing (geometric)
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    """
    Gebruik 30d 1h OHLCV; neem P10–P90 van close. Fall back: median ± BAND_PCT.
    """
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
    # fallback op recente prijs
    last = float(ex.fetch_ticker(pair)["last"])
    low  = last * (1 - BAND_PCT)
    high = last * (1 + BAND_PCT)
    return low, high

# ------------------ Strategie state ------------------
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    # EUR cash wordt verdeeld volgens weights; coins starten op 0
    port = {"EUR": CAPITAL_EUR, "coins": {}, "pnl_realized": 0.0}
    for p in pairs:
        port["coins"][p] = {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]}
    return port

def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    if weights_csv:
        parts = [x.strip() for x in weights_csv.split(",") if x.strip()]
        d = {}
        for p in parts:
            k, v = p.split(":")
            d[k.strip()] = float(v)
        # normalize
        s = sum(d.get(x, 0.0) for x in pairs)
        if s > 0:
            return {x: d.get(x, 0.0)/s for x in pairs}
    # default: equal weight
    eq = 1.0 / len(pairs)
    return {x: eq for x in pairs}

def mk_grid_state(ex, pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(ex, pair)
    grid_levels = geometric_levels(low, high, levels)
    return {
        "low": low, "high": high, "levels": grid_levels,
        "last_price": None,
        # filled buy "tickets": list of dicts with qty & buy_price
        "inventory_lots": [],   # [{qty, buy_price}]
    }

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    total = state["portfolio"]["EUR"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

# ------------------ Fillsimulator ------------------
def try_fill_grid(pair: str, price_now: float, price_prev: float, grid: dict, port: dict, ex) -> List[str]:
    """
    Detecteert crosses van grid levels tussen price_prev -> price_now.
    Buy: neerwaarts crossing, Sell: opwaarts crossing (voor bestaande lot).
    Returns: list of logregels (voor stdout).
    """
    logs = []
    levels = grid["levels"]

    # EUR beschikbaar voor dit pair
    cash_alloc = port["coins"][pair]["cash_alloc"]
    # Order size: deel allocatie over levels. Houd buffer aan (10%) om fees & misfills op te vangen.
    order_eur = max(1.0, (cash_alloc * 0.90) / (len(levels)//2))

    # BUY: cross van boven -> beneden door een level
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            if port["portfolio_eur"] <= 0.5 or order_eur > port["portfolio_eur"]:
                logs.append(f"[{pair}] BUY skip: onvoldoende EUR")
                continue
            qty = order_eur / L
            fee_eur = order_eur * FEE_PCT
            port["portfolio_eur"] -= (order_eur + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": L})
            realized = 0.0
            append_csv(TRADES_CSV, [now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                                    f"{port['portfolio_eur']:.2f}", pair.split("/")[0],
                                    f"{port['coins'][pair]['qty']:.8f}", f"{realized:.2f}", "grid_buy"])
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{L:.6f} | EUR_cash={port['portfolio_eur']:.2f}")

    # SELL: cross van beneden -> boven; verkoop FIFO lot met markup ~grid step
    if price_prev is not None and price_now > price_prev and grid["inventory_lots"]:
        crossed = [L for L in levels if price_prev < L <= price_now]
        for L in crossed:
            # verkoop kleinste lot waarvan L > buy_price (winst)
            # (Eenvoudig FIFO; je kunt ook best-match zoeken)
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
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{L:.6f} | PnL={pnl:.2f} | EUR_cash={port['portfolio_eur']:.2f}")

    grid["last_price"] = price_now
    return logs

# ------------------ Main loop ------------------
def main():
    print(f"== PAPER GRID BOT start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% ==")
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    weights = normalize_weights(pairs, WEIGHTS_CSV)

    ex = make_ex()
    # filter paren die niet bestaan
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden op exchange.")

    state = load_state() or {}
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
        state["portfolio"]["portfolio_eur"] = CAPITAL_EUR  # standalone cash pot
        state["pnl_realized_total"] = 0.0

    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    save_state(state)

    # Dagelijkse equity-snapshot
    last_equity_day = None

    while True:
        try:
            # snapshot equity 1x per dag
            today = datetime.now(timezone.utc).date().isoformat()
            if today != last_equity_day:
                equity = mark_to_market(ex, state, pairs)
                append_csv(EQUITY_CSV, [today, f"{equity:.2f}"])
                last_equity_day = today

            # per pair prijs ophalen en eventueel fills simuleren
            for p in pairs:
                t = ex.fetch_ticker(p)
                px = float(t["last"])
                grid = state["grids"][p]
                logs = try_fill_grid(p, px, grid["last_price"], grid, state["portfolio"], ex)
                if logs:
                    print("\n".join(logs))

            save_state(state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[neterr] {e}; backoff..")
            time.sleep(2 + random.random())
        except Exception as e:
            # zorg dat de worker blijft leven
            print(f"[runtime] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
