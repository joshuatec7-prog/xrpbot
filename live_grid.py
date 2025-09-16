# live_grid.py
# --- Multi-coin LIVE grid bot (Bitvavo) ---
# - Beschermt bestaande wallet (verkoopt ALLEEN bot-lots)
# - Speelpot via CAPITAL_EUR (default 1000) met harde cap
# - Verkoop ALLEEN bij netto winst (MIN_PROFIT_PCT óf MIN_PROFIT_EUR)
# - CSV logging: data/live_trades.csv, data/live_equity.csv
# - Console samenvattingen elke LOG_SUMMARY_SEC en rapport elke REPORT_EVERY_HOURS

import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ---------- helpers ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

def append_csv(path: Path, row: List[str], header: List[str] | None = None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header:
            w.writerow(header)
        w.writerow(row)

def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def save_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

# ---------- ENV / Defaults (alfabetisch) ----------
API_KEY      = os.getenv("API_KEY", "")
API_SECRET   = os.getenv("API_SECRET", "")
BAND_PCT     = float(os.getenv("BAND_PCT", "0.20"))
CAPITAL_EUR  = float(os.getenv("CAPITAL_EUR", "1000"))
COINS_CSV    = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
DATA_DIR_ARG = os.getenv("DATA_DIR", "data")
EXCHANGE_ID  = os.getenv("EXCHANGE", "bitvavo").strip().lower()
FEE_PCT      = float(os.getenv("FEE_PCT", "0.0015"))
GRID_LEVELS  = int(os.getenv("GRID_LEVELS", "24"))
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE","true").lower() in ("1","true","yes")
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC", "240"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
MIN_PROFIT_EUR = float(os.getenv("MIN_PROFIT_EUR", "0.25"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.006"))
OPERATOR_ID  = os.getenv("OPERATOR_ID", "").strip()
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR", "2.0"))
REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "4"))
SKIM_INTERVAL_MIN = int(os.getenv("SKIM_INTERVAL_MIN", "15"))
SKIM_MIN_EUR = float(os.getenv("SKIM_MIN_EUR", "100"))
SKIM_PROFITS = os.getenv("SKIM_PROFITS", "false").lower() in ("1","true","yes")
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC","300"))
SLEEP_SEC     = int(os.getenv("SLEEP_SEC", "15"))
WEIGHTS_CSV   = os.getenv("WEIGHTS","BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()

if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY / API_SECRET ontbreken (Render → Environment).")

# ---------- storage (Render-vriendelijk) ----------
try:
    DATA_DIR = Path(DATA_DIR_ARG)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    DATA_DIR = Path("./data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE  = DATA_DIR / "live_state.json"
TRADES_CSV  = DATA_DIR / "live_trades.csv"
EQUITY_CSV  = DATA_DIR / "live_equity.csv"

# ---------- exchange ----------
def make_exchange():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })
    if OPERATOR_ID:
        ex.options["operatorId"] = OPERATOR_ID
    ex.load_markets()
    return ex

# ---------- wegingen ----------
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
    s = sum(d.values())
    if s <= 0:
        eq = 1.0 / len(pairs) if pairs else 0.0
        return {p: eq for p in pairs}
    return {p: (d.get(p, 0.0) / s) for p in pairs}

# ---------- grid bouw ----------
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    """
    P10-P90 van 1h closes (30d); fallback ±BAND_PCT rond last.
    """
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
    return last * (1 - BAND_PCT), last * (1 + BAND_PCT)

def mk_grid_state(ex, pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(ex, pair)
    return {
        "pair": pair,
        "low": low,
        "high": high,
        "levels": geometric_levels(low, high, levels),
        "last_price": None,
        "inventory_lots": [],   # [{qty, buy_price}]
    }

# ---------- portfolio ----------
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    return {
        "play_cash_eur": CAPITAL_EUR,     # alleen speelpot
        "cash_eur": CAPITAL_EUR,
        "pnl_realized": 0.0,
        "coins": {p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs},
        "last_skim_ts": 0.0,
    }

def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    if n_levels < 2:
        n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)  # 90% alloc in helft van levels
    return max(5.0, base * ORDER_SIZE_FACTOR)

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    total = state["portfolio"]["cash_eur"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

def portfolio_exposure_eur(ex, port: dict, pairs: List[str]) -> float:
    expo = 0.0
    for p in pairs:
        qty = port["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            expo += qty * px
    return expo

# ---------- winst en fills ----------
def net_gain_ok(buy_price: float, sell_avg: float, fee_pct: float,
                min_pct: float, min_eur: float, qty: float) -> bool:
    if buy_price <= 0 or sell_avg <= 0 or qty <= 0:
        return False
    gross_pct = (sell_avg - buy_price) / buy_price
    net_pct   = gross_pct - 2.0 * fee_pct
    net_eur   = (sell_avg - buy_price) * qty - (sell_avg * qty) * fee_pct - (buy_price * qty) * fee_pct
    return (net_pct >= min_pct) or (net_eur >= min_eur)

def buy_market(ex, pair: str, eur: float) -> Tuple[float, float, float]:
    """
    Koop voor 'eur' EUR; return (filled_qty, avg_price, fee_eur).
    """
    if eur < 5.0:
        return 0.0, 0.0, 0.0
    params = {"cost": float(f"{eur:.2f}")}  # Bitvavo: market buy by cost
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    order = ex.create_order(pair, "market", "buy", None, None, params)
    avgp = float(order.get("average") or order.get("price") or 0.0)
    filled = float(order.get("filled") or order.get("info", {}).get("filledAmount") or 0.0)
    fee_eur = eur * FEE_PCT
    return filled, avgp, fee_eur

def sell_market(ex, pair: str, qty: float) -> Tuple[float, float, float]:
    """
    Verkoop qty; return (proceeds_eur, avg_price, fee_eur).
    """
    if qty <= 0:
        return 0.0, 0.0, 0.0
    params = {}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    order = ex.create_order(pair, "market", "sell", qty, None, params)
    avgp = float(order.get("average") or order.get("price") or 0.0)
    proceeds = avgp * qty
    fee_eur = proceeds * FEE_PCT
    return proceeds, avgp, fee_eur

# ---------- grid logica ----------
def try_grid_live(ex, pair: str, price_now: float, price_prev: float,
                  grid: dict, port: dict, pairs_all: List[str]) -> List[str]:
    levels = grid["levels"]
    logs: List[str] = []
    if not levels:
        return logs

    # BUY: neerwaartse cross -> koop ticket als cash én exposure het toelaten
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now < L <= price_prev]
        for _ in crossed:
            ticket_eur = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels))

            # harde cap: nooit boven speelpot
            exposure = portfolio_exposure_eur(ex, port, pairs_all)
            if (port["cash_eur"] - MIN_CASH_BUFFER_EUR) < ticket_eur:
                logs.append(f"[{pair}] BUY skip: onvoldoende cash (buffer).")
                continue
            if (port["cash_eur"] + exposure) >= (CAPITAL_EUR - 1e-6):
                logs.append(f"[{pair}] BUY skip: capital cap bereikt.")
                continue

            qty, avgp, fee_eur = buy_market(ex, pair, ticket_eur)
            if qty <= 0 or avgp <= 0:
                continue

            grid["inventory_lots"].append({"qty": qty, "buy_price": avgp})
            port["cash_eur"] -= (ticket_eur + fee_eur)
            port["coins"][pair]["qty"] += qty

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "BUY", f"{avgp:.6f}", f"{qty:.8f}", f"{ticket_eur:.2f}", f"{port['cash_eur']:.2f}",
                 pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
                header=["timestamp","pair","side","avg_price","qty","eur","cash_eur",
                        "base","base_qty","pnl_eur","comment"]
            )
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{avgp:.6f} | cost≈€{ticket_eur:.2f} | fee≈€{fee_eur:.2f} | cash=€{port['cash_eur']:.2f}")

    # SELL: winst-gestuurd per lot, onafhankelijk van level-cross
    if grid["inventory_lots"]:
        sell_idx = None
        for i, lot in enumerate(grid["inventory_lots"]):
            if net_gain_ok(lot["buy_price"], price_now, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, lot["qty"]):
                sell_idx = i
                break

        if sell_idx is not None:
            lot = grid["inventory_lots"][sell_idx]
            qty = lot["qty"]
            proceeds, avgp, fee_eur = sell_market(ex, pair, qty)
            if proceeds > 0 and avgp > 0 and net_gain_ok(lot["buy_price"], avgp, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, qty):
                grid["inventory_lots"].pop(sell_idx)
                pnl = proceeds - fee_eur - (qty * lot["buy_price"])
                port["cash_eur"] += (proceeds - fee_eur)
                port["coins"][pair]["qty"] -= qty
                port["pnl_realized"] += pnl

                append_csv(
                    TRADES_CSV,
                    [now_iso(), pair, "SELL", f"{avgp:.6f}", f"{qty:.8f}", f"{proceeds:.2f}", f"{port['cash_eur']:.2f}",
                     pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "take_profit"]
                )
                logs.append(
                    f"[{pair}] SELL {qty:.8f} @ €{avgp:.6f} | proceeds=€{proceeds:.2f} | fee=€{fee_eur:.2f} "
                    f"| pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}"
                )

    grid["last_price"] = price_now
    return logs

# ---------- main ----------
def main():
    # pairs + weights
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    ex = make_exchange()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)

    # state laden
    if STATE_FILE.exists():
        state = load_json(STATE_FILE, {})
    else:
        state = {}

    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    # baseline note (alleen informatief)
    if LOCK_PREEXISTING_BALANCE and "baseline" not in state:
        bal = ex.fetch_balance().get("free", {})
        state["baseline"] = {p.split("/")[0]: float(bal.get(p.split("/")[0], 0) or 0.0) for p in pairs}

    save_json(STATE_FILE, state)

    print(
        f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={pct(FEE_PCT)} | "
        f"pairs={pairs} | weights={weights} | factor={ORDER_SIZE_FACTOR} |"
        f" min_profit={pct(MIN_PROFIT_PCT)} / €{MIN_PROFIT_EUR:.2f}"
    )

    last_report_ts = 0.0
    last_summary_ts = 0.0

    while True:
        try:
            # heartbeat + equity log
            eq = mark_to_market(ex, state, pairs)
            append_csv(
                EQUITY_CSV,
                [datetime.now(timezone.utc).date().isoformat(), f"{eq:.2f}"],
                header=["date", "total_equity_eur"]
            )

            # korte console summary
            now_ts = time.time()
            if now_ts - last_summary_ts >= LOG_SUMMARY_SEC:
                since_start = eq - CAPITAL_EUR
                col_start = "\033[92m" if since_start >= 0 else "\033[91m"
                col_end = "\033[0m"
                pr = state["portfolio"]["pnl_realized"]
                cash = state["portfolio"]["cash_eur"]
                print(
                    f"[SUMMARY] total_eq=€{eq:.2f} | cash=€{cash:.2f} | pnl_realized=€{pr:.2f} | "
                    f"since_start={col_start}{since_start:.2f}{col_end}"
                )
                last_summary_ts = now_ts

            # per pair: prijs ophalen en grid logica doen
            for p in pairs:
                t = ex.fetch_ticker(p)
                px = float(t["last"])
                grid = state["grids"][p]
                logs = try_grid_live(ex, p, px, grid["last_price"], grid, state["portfolio"], pairs)
                if logs:
                    print("\n".join(logs))

            # rapport (console) om de X uur
            if time.time() - last_report_ts >= REPORT_EVERY_HOURS * 3600:
                pr = state["portfolio"]["pnl_realized"]
                cash = state["portfolio"]["cash_eur"]
                since_start = eq - CAPITAL_EUR
                col_start = "\033[92m" if since_start >= 0 else "\033[91m"
                col_end = "\033[0m"
                print(
                    f"[REPORT] total_eq=€{eq:.2f} | cash=€{cash:.2f} | pnl_realized=€{pr:.2f} | "
                    f"since_start={col_start}{since_start:.2f}{col_end} | pairs={pairs}"
                )
                last_report_ts = time.time()

            save_json(STATE_FILE, state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[net] {e}; backoff..")
            time.sleep(2 + random.random())
        except ccxt.BaseError as e:
            print(f"[ccxt] {e}; wacht..")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Gestopt.")
            break
        except Exception as e:
            print(f"[runtime] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
