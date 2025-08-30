# paper_grid.py
# 24/7 PAPER GRID (paper trading; publieke prijzen via ccxt; GEEN echte orders)
# - Grid per pair op basis van P10–P90 van 30d 1h data (fallback: median ± BAND_PCT)
# - Virtuele fills met fees, FIFO inventory, persistente state
# - Logt elke trade + periodieke SUMMARY naar stdout (Render Live Logs)
# - Schrijft trades.csv, equity.csv en state.json naar DATA_DIR

import os, json, time, math, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ------------------ Config via ENV ------------------
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "15000"))

# COINS als CSV: "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
COINS_CSV = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR")

# Weights als CSV "PAIR:WEIGHT,...", bv "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10"
WEIGHTS_CSV = os.getenv("WEIGHTS", "").strip()

# Grid & fees
GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))     # fallback band ±20%
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))    # 0.15% per order
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "30"))      # loop interval

# Hoe vaak een grote samenvatting in de logs?
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC", "600"))  # elke 10 min

# Waar bestanden opslaan (zet optioneel ENV DATA_DIR=/data als je een Render Disk mount)
DATA_DIR = Path(os.getenv("DATA_DIR", "."))
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE  = DATA_DIR / "state.json"
TRADES_CSV  = DATA_DIR / "trades.csv"
EQUITY_CSV  = DATA_DIR / "equity.csv"

EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")  # alleen publiek voor prijzen

# ------------------ Helpers ------------------
def now_iso_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

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
                w.writerow(["timestamp","pair","side","price","amount","fee_eur","cash_eur","base","base_qty","realized_pnl_eur","comment"])
            elif path == EQUITY_CSV:
                w.writerow(["timestamp","total_equity_eur","cash_eur","pnl_realized_eur"])
        w.writerow(row)

# ------------------ Exchange (public) ------------------
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
    """
    Gebruik 30d 1h OHLCV; neem P10–P90 van close. Fall back: last ± BAND_PCT.
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
    last = float(ex.fetch_ticker(pair)["last"])
    low  = last * (1 - BAND_PCT)
    high = last * (1 + BAND_PCT)
    return low, high

# ------------------ Portfolio & weights ------------------
def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    if weights_csv:
        d = {}
        for p in [x.strip() for x in weights_csv.split(",") if x.strip()]:
            k, v = p.split(":")
            d[k.strip().upper()] = float(v)
        s = sum(d.get(x, 0.0) for x in pairs)
        if s > 0:
            return {x: d.get(x, 0.0)/s for x in pairs}
    eq = 1.0 / len(pairs)
    return {x: eq for x in pairs}

def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    # EUR cash is één pot (portfolio_eur). cash_alloc per pair bepaalt BUY sizedeur.
    port = {"portfolio_eur": CAPITAL_EUR, "coins": {}, "pnl_realized": 0.0}
    for p in pairs:
        port["coins"][p] = {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]}
    return port

# ------------------ Grid state per pair ------------------
def mk_grid_state(ex, pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(ex, pair)
    grid_levels = geometric_levels(low, high, levels)
    return {
        "low": low, "high": high, "levels": grid_levels,
        "last_price": None,
        "inventory_lots": [],   # [{qty, buy_price}]
    }

# ------------------ MTM / equity ------------------
def mark_to_market(ex, state: dict, pairs: List[str]) -> Tuple[float, float]:
    total = state["portfolio"]["portfolio_eur"]
    coins_val = 0.0
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            coins_val += qty * px
    return total + coins_val, coins_val

# ------------------ Fillsimulator (crossings) ------------------
def try_fill_grid(pair: str, price_now: float, price_prev: float, grid: dict, port: dict, ex) -> List[str]:
    """
    BUY: neerwaarts crossing, SELL: opwaarts crossing (FIFO lot).
    EUR ordergrootte: 90% van cash_alloc gedeeld door helft van levels.
    """
    logs = []
    levels = grid["levels"]
    base = pair.split("/")[0]

    cash_alloc = port["coins"][pair]["cash_alloc"]
    order_eur = max(1.0, (cash_alloc * 0.90) / max(1, (len(levels)//2)))

    # BUY cross (boven->beneden)
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            if port["portfolio_eur"] <= (order_eur + order_eur*FEE_PCT):
                logs.append(f"[{pair}] BUY skip: onvoldoende EUR (vrij €{port['portfolio_eur']:.2f})")
                continue
            qty = order_eur / L
            fee_eur = order_eur * FEE_PCT
            port["portfolio_eur"] -= (order_eur + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": L})
            append_csv(TRADES_CSV, [now_iso_local(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                                    f"{port['portfolio_eur']:.2f}", base, f"{port['coins'][pair]['qty']:.8f}",
                                    f"{0.0:.2f}", "grid_buy"])
            logs.append(f"[PAPER] {pair} BUY {qty:.6f} {base} @ €{L:.2f} | fee≈€{fee_eur:.2f} | cash=€{port['portfolio_eur']:.2f}")

    # SELL cross (beneden->boven); verkoop eerste lot met winst
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
            append_csv(TRADES_CSV, [now_iso_local(), pair, "SELL", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                                    f"{port['portfolio_eur']:.2f}", base, f"{port['coins'][pair]['qty']:.8f}",
                                    f"{pnl:.2f}", "grid_sell"])
            logs.append(f"[PAPER] {pair} SELL {qty:.6f} {base} @ €{L:.2f} | pnl=€{pnl:.2f} | fee≈€{fee_eur:.2f} | cash=€{port['portfolio_eur']:.2f}")

    grid["last_price"] = price_now
    return logs

# ------------------ Main loop ------------------
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    ex = make_ex()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden op exchange.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)

    # State
    state = load_state() or {}
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    print(f"== PAPER GRID START | capital=€{CAPITAL_EUR:.2f} | fee={FEE_PCT*100:.3f}% | levels={GRID_LEVELS} ==")
    print(f"Pairs: {pairs}")
    print(f"Weights: { {p: round(weights[p], 3) for p in pairs} }")
    print(f"Data dir: {DATA_DIR.resolve()}  (trades.csv, equity.csv, state.json)")
    save_state(state)

    last_summary_at = 0.0

    while True:
        try:
            # Periodieke SUMMARY
            now = time.time()
            if now - last_summary_at >= LOG_SUMMARY_SEC:
                eq, coins_val = mark_to_market(ex, state, pairs)
                cash = state["portfolio"]["portfolio_eur"]
                pnl  = state["portfolio"]["pnl_realized"]
                append_csv(EQUITY_CSV, [now_iso_local(), f"{eq:.2f}", f"{cash:.2f}", f"{pnl:.2f}"])
                print(f"[SUMMARY] equity=€{eq:.2f} | cash=€{cash:.2f} | coins=€{coins_val:.2f} | pnl_realized=€{pnl:.2f}")
                for p in pairs:
                    px = float(ex.fetch_ticker(p)["last"])
                    qty = state["portfolio"]["coins"][p]["qty"]
                    print(f"         - {p}: px=€{px:.2f} qty={qty:.6f}")
                last_summary_at = now

            # Per pair prijs ophalen en crossings verwerken
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
            print(f"[runtime] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
