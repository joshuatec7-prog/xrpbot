# paper_grid.py — Multi-coin Paper Grid Bot met LONG + (virtuele) SHORT
# - COINS & WEIGHTS uit ENV
# - Gridband 30d 1h P10–P90 (fallback median ± BAND_PCT)
# - Virtuele fills: BUY/SELL (long) en SELL_SHORT/BUY_TO_COVER (short)
# - Persistente state + trades.csv + equity.csv (in DATA_DIR)
# - ORDER_SIZE_FACTOR om ticketgrootte te tunen (agressiever)
# - SHORTS: margin lock & correcte equity (liability) verwerking

import os, json, time, math, csv, random, statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ------------------ ENV / Defaults ------------------
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "15000"))

COINS_CSV = os.getenv(
    "COINS",
    "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
).strip()

WEIGHTS_CSV = os.getenv(
    "WEIGHTS",
    "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10"
).strip()

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "30"))
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")
DATA_DIR    = Path(os.getenv("DATA_DIR", "/var/data"))

# Order sizing
ORDER_SIZE_FACTOR     = float(os.getenv("ORDER_SIZE_FACTOR", "2.0"))

# Veiligheid
MIN_CASH_BUFFER_EUR   = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
MIN_TRADE_EUR         = float(os.getenv("MIN_TRADE_EUR", "5"))

LOG_SUMMARY_SEC       = int(os.getenv("LOG_SUMMARY_SEC", "600"))

# Shorts (virtueel, paper only)
ENABLE_SHORTS         = os.getenv("ENABLE_SHORTS", "false").lower() in ("1","true","yes")
SHORT_ALLOC_PCT       = float(os.getenv("SHORT_ALLOC_PCT", "0.30"))  # aandeel van per-coin alloc voor shorts
LONG_ALLOC_PCT_ENV    = os.getenv("LONG_ALLOC_PCT", "")              # optioneel; anders = 1 - SHORT_ALLOC_PCT
LONG_ALLOC_PCT        = float(LONG_ALLOC_PCT_ENV) if LONG_ALLOC_PCT_ENV else max(0.0, 1.0 - SHORT_ALLOC_PCT)
MARGIN_FACTOR         = float(os.getenv("MARGIN_FACTOR", "0.5"))     # % van notional te locken als margin

# ------------------ Bestandslocaties ------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
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
                w.writerow([
                    "timestamp","pair","side","price","amount",
                    "fee_eur","cash_eur","coin","coin_qty","realized_pnl_eur","comment"
                ])
            elif path == EQUITY_CSV:
                w.writerow(["date","total_equity_eur"])
        w.writerow(row)

def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    if weights_csv:
        for item in [x.strip() for x in weights_csv.split(",") if x.strip()]:
            if ":" in item:
                k, v = item.split(":", 1)
                try:
                    d[k.strip().upper()] = float(v)
                except Exception:
                    pass
    d = {p: d.get(p, 0.0) for p in pairs}
    s = sum(d.values())
    if s > 0:
        return {p: (d[p] / s) for p in pairs}
    eq = 1.0 / len(pairs) if pairs else 0.0
    return {p: eq for p in pairs}

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
    """P10–P90 van 30d 1h closes; fallback median ± BAND_PCT."""
    try:
        ohlcv = ex.fetch_ohlcv(pair, timeframe="1h", limit=24*30)
        if ohlcv and len(ohlcv) >= 50:
            closes = [c[4] for c in ohlcv if c and c[4] is not None]
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

def mk_grid_state(ex, pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(ex, pair)
    grid_levels = geometric_levels(low, high, levels)
    return {
        "pair": pair,
        "low": low,
        "high": high,
        "levels": grid_levels,
        "last_price": None,
        "inventory_lots": [],     # LONG lots: [{qty, buy_price}]
        "short_lots": [],         # SHORT lots: [{qty, sell_price, fee_open, margin}]
    }

# ------------------ Portfolio & waardering ------------------
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    return {
        "portfolio_eur": CAPITAL_EUR,    # centrale EUR pot (cash)
        "pnl_realized": 0.0,
        "short_margin_locked": 0.0,      # gereserveerde margin voor shorts
        "coins": {
            p: {
                "qty": 0.0,              # long qty
                "short_qty": 0.0,        # outstanding short (som lots)
                "cash_alloc": CAPITAL_EUR * weights[p]
            } for p in pairs
        }
    }

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    """
    Equity = cash + long_value - short_liability (qty_short * prijs).
    """
    total = state["portfolio"]["portfolio_eur"]
    for p in pairs:
        px = float(ex.fetch_ticker(p)["last"])
        long_qty  = state["portfolio"]["coins"][p]["qty"]
        short_qty = state["portfolio"]["coins"][p]["short_qty"]
        if long_qty > 0:
            total += long_qty * px
        if short_qty > 0:
            total -= short_qty * px
    return total

# ------------------ Order sizing ------------------
def euro_per_ticket(alloc_eur: float, n_levels: int) -> float:
    """
    Basisticket: ~90% van (alloc_eur) / (n_levels/2) * ORDER_SIZE_FACTOR
    """
    if n_levels < 2:
        n_levels = 2
    base = (alloc_eur * 0.90) / (n_levels // 2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

# ------------------ Fillsimulator ------------------
def try_fill_grid(pair: str, price_now: float, price_prev: float,
                  grid: dict, port: dict, ex) -> List[str]:
    logs = []
    levels = grid["levels"]

    cash_alloc = port["coins"][pair]["cash_alloc"]
    order_eur_long  = euro_per_ticket(cash_alloc * LONG_ALLOC_PCT,  len(levels))
    order_eur_short = euro_per_ticket(cash_alloc * SHORT_ALLOC_PCT, len(levels)) if ENABLE_SHORTS and SHORT_ALLOC_PCT > 0 else 0.0

    def free_eur() -> float:
        # cash minus locked short margin en buffer
        return port["portfolio_eur"] - port.get("short_margin_locked", 0.0) - MIN_CASH_BUFFER_EUR

    # ---------------- LONG ----------------
    # BUY: neerwaartse cross
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            if order_eur_long <= 0:
                continue
            if free_eur() < (order_eur_long + order_eur_long * FEE_PCT):
                logs.append(f"[{pair}] BUY skip: onvoldoende vrije EUR (incl. buffer/margin).")
                continue
            qty = order_eur_long / L
            fee_eur = order_eur_long * FEE_PCT
            port["portfolio_eur"] -= (order_eur_long + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": L})
            append_csv(TRADES_CSV, [
                now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}",
                f"{fee_eur:.2f}", f"{port['portfolio_eur']:.2f}",
                pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}",
                f"{0.0:.2f}", "grid_buy"
            ])
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{L:.6f} | cash=€{port['portfolio_eur']:.2f}")

    # SELL: opwaartse cross (realiseer long-lot met winst)
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
            append_csv(TRADES_CSV, [
                now_iso(), pair, "SELL", f"{L:.6f}", f"{qty:.8f}",
                f"{fee_eur:.2f}", f"{port['portfolio_eur']:.2f}",
                pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}",
                f"{pnl:.2f}", "grid_sell"
            ])
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{L:.6f} | PnL={pnl:.2f} | cash=€{port['portfolio_eur']:.2f}")

    # ---------------- SHORT (virtueel) ----------------
    if ENABLE_SHORTS and SHORT_ALLOC_PCT > 0:
        # SELL_SHORT (open): opwaartse cross
        if price_prev is not None and price_now > price_prev:
            crossed = [L for L in levels if price_prev < L <= price_now]
            for L in crossed:
                if order_eur_short <= 0:
                    continue
                margin_req = order_eur_short * MARGIN_FACTOR
                if free_eur() < margin_req:
                    logs.append(f"[{pair}] SHORT skip (open): onvoldoende vrije EUR voor margin.")
                    continue
                qty = order_eur_short / L
                proceeds = qty * L
                fee_open = proceeds * FEE_PCT
                # bij short ontvang je proceeds (minus fee), maar reserveer margin
                port["portfolio_eur"] += (proceeds - fee_open)
                port["short_margin_locked"] += margin_req
                port["coins"][pair]["short_qty"] += qty
                grid["short_lots"].append({"qty": qty, "sell_price": L, "fee_open": fee_open, "margin": margin_req})
                append_csv(TRADES_CSV, [
                    now_iso(), pair, "SELL_SHORT", f"{L:.6f}", f"{qty:.8f}",
                    f"{fee_open:.2f}", f"{port['portfolio_eur']:.2f}",
                    pair.split("/")[0], f"-{port['coins'][pair]['short_qty']:.8f}",
                    f"{0.0:.2f}", "grid_short_open"
                ])
                logs.append(f"[{pair}] SELL_SHORT {qty:.8f} @ €{L:.6f} | cash=€{port['portfolio_eur']:.2f} | margin_locked=€{port['short_margin_locked']:.2f}")

        # BUY_TO_COVER (close): neerwaartse cross
        if price_prev is not None and price_now < price_prev and grid["short_lots"]:
            crossed = [L for L in levels if price_now <= L < price_prev]
            for L in crossed:
                lot_idx = None
                for i, lot in enumerate(grid["short_lots"]):
                    if L < lot["sell_price"]:   # winstgevend cover
                        lot_idx = i
                        break
                if lot_idx is None:
                    # desnoods niets doen (HOLD) tot winst
                    continue
                lot = grid["short_lots"].pop(lot_idx)
                qty = lot["qty"]
                cost = qty * L
                fee_close = cost * FEE_PCT
                pnl = (lot["sell_price"] - L) * qty - (lot["fee_open"] + fee_close)
                # bij cover betaal je cost + fee; margin vrijgeven
                port["portfolio_eur"] -= (cost + fee_close)
                port["short_margin_locked"] -= lot["margin"]
                port["coins"][pair]["short_qty"] -= qty
                port["pnl_realized"] += pnl
                append_csv(TRADES_CSV, [
                    now_iso(), pair, "BUY_TO_COVER", f"{L:.6f}", f"{qty:.8f}",
                    f"{fee_close:.2f}", f"{port['portfolio_eur']:.2f}",
                    pair.split("/")[0], f"-{port['coins'][pair]['short_qty']:.8f}",
                    f"{pnl:.2f}", "grid_short_close"
                ])
                logs.append(f"[{pair}] BUY_TO_COVER {qty:.8f} @ €{L:.6f} | PnL={pnl:.2f} | cash=€{port['portfolio_eur']:.2f} | margin_locked=€{port['short_margin_locked']:.2f}")

    grid["last_price"] = price_now
    return logs

# ------------------ Main ------------------
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    weights = normalize_weights(pairs, WEIGHTS_CSV)

    ex = make_ex()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden op exchange.")

    print(f"== PAPER GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} "
          f"| fee={FEE_PCT*100:.3f}% | pairs={pairs} | weights={weights} "
          f"| factor={ORDER_SIZE_FACTOR} | shorts={'ON' if ENABLE_SHORTS else 'OFF'} "
          f"(long_alloc={LONG_ALLOC_PCT:.2f}, short_alloc={SHORT_ALLOC_PCT:.2f}, margin={MARGIN_FACTOR:.2f})")

    state = load_state() or {}
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    save_state(state)

    last_equity_day = None
    last_sum = 0.0

    while True:
        try:
            # Dagelijkse equity snapshot
            today = datetime.now(timezone.utc).date().isoformat()
            if today != last_equity_day:
                equity = mark_to_market(ex, state, pairs)
                append_csv(EQUITY_CSV, [today, f"{equity:.2f}"])
                last_equity_day = today

            # Per pair prijs ophalen en fills simuleren
            for p in pairs:
                t = ex.fetch_ticker(p)
                px = float(t["last"])
                grid = state["grids"][p]
                logs = try_fill_grid(p, px, grid["last_price"], grid, state["portfolio"], ex)
                if logs:
                    print("\n".join(logs))

            # Periodieke samenvatting
            now = time.time()
            if now - last_sum >= LOG_SUMMARY_SEC:
                eq = mark_to_market(ex, state, pairs)
                cash = state["portfolio"]["portfolio_eur"]
                pr = state["portfolio"]["pnl_realized"]
                short_lock = state["portfolio"].get("short_margin_locked", 0.0)
                print(f"[SUMMARY] equity=€{eq:.2f} | cash=€{cash:.2f} | pnl_realized=€{pr:.2f} | margin_locked=€{short_lock:.2f}")
                last_sum = now

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
