# paper_grid.py — Multi-coin Paper Grid Bot met "profit skim"
# - COINS & WEIGHTS uit ENV
# - Gridband via 30d/1h P10–P90 (fallback median ± BAND_PCT)
# - Virtuele fills: BUY/SELL (paper), fees & realized PnL
# - Persistente state + trades.csv + equity.csv (in DATA_DIR)
# - ORDER_SIZE_FACTOR bepaalt ticketgrootte
# - RESET_STATE/WARM_START worden gerespecteerd
# - NIEUW: houdt trading-equity rond CAPITAL_EUR en roomt overtollige cash af naar profit-pot

import os, json, time, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ------------------ ENV / Defaults ------------------
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "50000"))

COINS_CSV = os.getenv(
    "COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
).strip()

WEIGHTS_CSV = os.getenv(
    "WEIGHTS", "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10"
).strip()

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))   # 0.15%
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "30"))
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")
DATA_DIR    = Path(os.getenv("DATA_DIR", "/var/data"))

ORDER_SIZE_FACTOR     = float(os.getenv("ORDER_SIZE_FACTOR", "1.0"))
MIN_CASH_BUFFER_EUR   = float(os.getenv("MIN_CASH_BUFFER_EUR", "250"))
MIN_TRADE_EUR         = float(os.getenv("MIN_TRADE_EUR", "5"))
LOG_SUMMARY_SEC       = int(os.getenv("LOG_SUMMARY_SEC", "600"))

# extra flags (mag blijven staan in je Render env)
RESET_STATE = os.getenv("RESET_STATE", "false").lower() in ("1", "true", "yes")
WARM_START  = os.getenv("WARM_START",  "true").lower() in ("1", "true", "yes")

# NIEUW: winst-afromen
SKIM_PROFITS       = os.getenv("SKIM_PROFITS", "true").lower() in ("1","true","yes")
SKIM_INTERVAL_MIN  = int(os.getenv("SKIM_INTERVAL_MIN", "10"))
SKIM_MIN_EUR       = float(os.getenv("SKIM_MIN_EUR", "50"))

# ------------------ Bestandslocaties ------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE  = DATA_DIR / "state.json"
TRADES_CSV  = DATA_DIR / "trades.csv"
EQUITY_CSV  = DATA_DIR / "equity.csv"

# ------------------ Helpers ------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

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
        "inventory_lots": [],   # [{qty, buy_price}]
    }

# ------------------ Portfolio & waardering ------------------
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    return {
        "portfolio_eur": CAPITAL_EUR,    # trading cash (rond 50k houden)
        "pnl_realized": 0.0,             # realized PnL van grid Sells
        "profit_eur": 0.0,               # AFGEROOMDE winst (niet meer mee-traden)
        "coins": {
            p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs
        }
    }

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    """Trading equity = trading cash + marktwaarde holdings (ZONDER profit-pot)."""
    total = state["portfolio"]["portfolio_eur"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

# ------------------ Order sizing ------------------
def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    """
    Basisticket: ~90% van allocatie / (n_levels/2) * ORDER_SIZE_FACTOR
    """
    if n_levels < 2:
        n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

# ------------------ Fillsimulator ------------------
def try_fill_grid(pair: str, price_now: float, price_prev: float,
                  grid: dict, port: dict, ex) -> List[str]:
    logs = []
    levels = grid["levels"]

    cash_alloc = port["coins"][pair]["cash_alloc"]
    order_eur  = euro_per_ticket(cash_alloc, len(levels))

    # BUY: neerwaartse cross
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            fee_eur = order_eur * FEE_PCT
            if port["portfolio_eur"] - MIN_CASH_BUFFER_EUR < (order_eur + fee_eur):
                logs.append(f"[{pair}] BUY skip: onvoldoende EUR (buffer).")
                continue
            if order_eur < MIN_TRADE_EUR:
                logs.append(f"[{pair}] BUY skip: onder min €{MIN_TRADE_EUR:.2f}.")
                continue

            qty = order_eur / L
            port["portfolio_eur"] -= (order_eur + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": L})

            append_csv(TRADES_CSV, [
                now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}",
                f"{fee_eur:.2f}", f"{port['portfolio_eur']:.2f}",
                pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}",
                f"{0.0:.2f}", "grid_buy"
            ])
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{L:.6f} | EUR_cash={port['portfolio_eur']:.2f}")

    # SELL: opwaartse cross
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
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{L:.6f} | PnL={pnl:.2f} | EUR_cash={port['portfolio_eur']:.2f}")

    grid["last_price"] = price_now
    return logs

# ------------------ Winst afromen ------------------
def skim_profits_if_needed(ex, state: dict, pairs: List[str], last_skim_ts: list):
    """Houd trading-equity ~ CAPITAL_EUR. Room overtollige cash af naar profit_eur."""
    if not SKIM_PROFITS:
        return

    now = time.time()
    if now - last_skim_ts[0] < SKIM_INTERVAL_MIN * 60:
        return

    port = state["portfolio"]
    cash = port["portfolio_eur"]

    # actuele holdings-waarde
    holdings_val = 0.0
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            holdings_val += qty * px

    # hoeveelheid cash nodig om totaal (cash+holdings) ≈ CAPITAL_EUR te houden
    needed_cash = max(0.0, CAPITAL_EUR - holdings_val) + MIN_CASH_BUFFER_EUR
    surplus = cash - needed_cash

    if surplus >= SKIM_MIN_EUR:
        port["portfolio_eur"] -= surplus
        port["profit_eur"]    += surplus
        print(f"[SKIM] afgeroomd €{surplus:.2f} → profit_pot; "
              f"cash=€{port['portfolio_eur']:.2f} | profit=€{port['profit_eur']:.2f}")
        last_skim_ts[0] = now

# ------------------ Main ------------------
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    weights = normalize_weights(pairs, WEIGHTS_CSV)

    ex = make_ex()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden op exchange.")

    print(f"== PAPER GRID start | capital=€{CAPITAL_EUR:.2f} |"
          f" levels={GRID_LEVELS} | fee=0.{int(FEE_PCT*1000):03d}% |"
          f" pairs={pairs} | weights={weights} | factor={ORDER_SIZE_FACTOR}"
          f" | buffer=€{MIN_CASH_BUFFER_EUR:.0f}")

    state = {}
    if RESET_STATE and not WARM_START:
        state = {}
        try:
            STATE_FILE.unlink(missing_ok=True)
        except Exception:
            pass
    else:
        state = load_state()

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
    last_skim_ts = [0.0]

    while True:
        try:
            # Dagelijkse equity snapshot (trading equity excl. profit-pot)
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

            # Winst afromen zodat trading-equity rond CAPITAL_EUR blijft
            skim_profits_if_needed(ex, state, pairs, last_skim_ts)

            # Periodieke samenvatting
            now = time.time()
            if now - last_sum >= LOG_SUMMARY_SEC:
                trading_eq = mark_to_market(ex, state, pairs)          # excl. profit-pot
                cash       = state["portfolio"]["portfolio_eur"]
                realized   = state["portfolio"]["pnl_realized"]
                skimmed    = state["portfolio"]["profit_eur"]
                total_net  = trading_eq + skimmed                      # trading + profit-pot

                print(f"[SUMMARY] trading_equity=€{trading_eq:.2f} | cash=€{cash:.2f} "
                      f"| pnl_realized=€{realized:.2f} | profit_pot=€{skimmed:.2f} "
                      f"| total_net=€{total_net:.2f} | target=€{CAPITAL_EUR:.2f}")
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
