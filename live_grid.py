# live_grid.py
# --- Multi-coin LIVE grid bot (Bitvavo/ccxt) ---
# - Weegt over meerdere pairs (BTC/EUR,ETH/EUR,...) met WEIGHTS
# - Beschermt bestaande balansen (LOCK_PREEXISTING_BALANCE)
# - Speelpot via CAPITAL_EUR (default 1000)
# - Schrijft state/trades/equity naar ./data (Render-vriendelijk)
# - Heartbeat in logs + 4u-rapport (kleur dagwinst)
# - Optioneel winst “skimmen” naar cash (niet herinvesteren)
# - Optioneel short (ENABLE_SHORT) en operatorId (Bitvavo)
# Pre-deploy tip (Render): mkdir -p /opt/render/project/src/data

import os, json, time, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ---------- Helpers ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def append_csv(path: Path, row: List, header: List[str]):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
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

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

# ---------- ENV / Defaults ----------
API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY / API_SECRET ontbreken (Render » .env.live).")

EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo").strip().lower()

CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "1000"))  # speelpot
COINS_CSV   = os.getenv("COINS",  "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
WEIGHTS_CSV = os.getenv("WEIGHTS","BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))

ORDER_SIZE_FACTOR   = float(os.getenv("ORDER_SIZE_FACTOR", "1.8"))
MIN_TRADE_EUR       = float(os.getenv("MIN_TRADE_EUR", "5"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))

REPORT_EVERY_HOURS  = float(os.getenv("REPORT_EVERY_HOURS", "4"))
SLEEP_SEC           = float(os.getenv("SLEEP_SEC", "15"))
HEARTBEAT_EVERY_SEC = int(os.getenv("HEARTBEAT_EVERY_SEC", "60"))
DISABLE_COLOR       = os.getenv("DISABLE_COLOR", "false").lower() in ("1","true","yes")

RESET_STATE = os.getenv("RESET_STATE", "false").lower() in ("1","true","yes")
WARM_START  = os.getenv("WARM_START",  "true").lower() in ("1","true","yes")

LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE","true").lower() in ("1","true","yes")
PROTECT_EXISTING         = os.getenv("PROTECT_EXISTING","true").lower() in ("1","true","yes")  # alias

SKIM_PROFITS   = os.getenv("SKIM_PROFITS","true").lower() in ("1","true","yes")
SKIM_MIN_EUR   = float(os.getenv("SKIM_MIN_EUR","100"))
SKIM_INTERVAL_MIN = int(os.getenv("SKIM_INTERVAL_MIN","15"))

ENABLE_SHORT          = os.getenv("ENABLE_SHORT","false").lower() in ("1","true","yes")
MAX_SHORT_EXPOSURE_FR = float(os.getenv("MAX_SHORT_EXPOSURE_FR","0.30"))

OPERATOR_ID = os.getenv("OPERATOR_ID","").strip()  # optioneel

# Storage
DATA_DIR = Path(os.getenv("DATA_DIR","data"))
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    DATA_DIR = Path("/tmp/live_grid_data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = DATA_DIR / "live_state.json"
TRADES_CSV = DATA_DIR / "live_trades.csv"
EQUITY_CSV = DATA_DIR / "live_equity.csv"

# ---------- Exchange ----------
def make_ex():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })
    if OPERATOR_ID:
        ex.options["operatorId"] = OPERATOR_ID  # Bitvavo vereist dit vaak
    ex.load_markets()
    return ex

# ---------- Weights ----------
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

# ---------- Grid-bouw ----------
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    """P10–P90 op 30d 1h; fallback ±BAND_PCT rond laatste prijs."""
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
    return {
        "pair": pair,
        "low": low,
        "high": high,
        "levels": geometric_levels(low, high, levels),
        "last_price": None,
        "inventory_lots": [],      # long lots: [{qty, buy_price}]
        "short_lots": [],          # short lots: [{qty, sell_price}] (alleen bij ENABLE_SHORT)
    }

# ---------- Portfolio ----------
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    return {
        "play_cash_eur": CAPITAL_EUR,            # referentie voor dag-change
        "cash_eur": CAPITAL_EUR,                 # vrije EUR binnen speelpot
        "pnl_realized": 0.0,                     # opgetelde gerealiseerde PnL (reset bij skim)
        "last_skim_ts": 0.0,
        "coins": {p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs},
        "short_exposure_eur": 0.0,               # totaal short exposure (ENABLE_SHORT)
    }

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    total = state["portfolio"]["cash_eur"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    if n_levels < 2:
        n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

# ---------- Skim profits ----------
def maybe_skim(state: dict):
    if not SKIM_PROFITS:
        return
    now = time.time()
    if now - state["portfolio"]["last_skim_ts"] < SKIM_INTERVAL_MIN * 60:
        return
    pr = state["portfolio"]["pnl_realized"]
    if pr >= SKIM_MIN_EUR and state["portfolio"]["cash_eur"] > pr:
        state["portfolio"]["cash_eur"] -= pr  # simuleer “uit speelpot”
        append_csv(
            TRADES_CSV,
            [now_iso(), "SKIM", "OUT", f"{pr:.2f}", "", "", f"{state['portfolio']['cash_eur']:.2f}", "skim_profits"],
            ["timestamp","pair","side","eur","price","amount","cash_eur","comment"]
        )
        state["portfolio"]["pnl_realized"] = 0.0
        state["portfolio"]["last_skim_ts"] = now

# ---------- LIVE market orders ----------
def buy_market(ex, pair: str, eur: float) -> Tuple[float, float]:
    if eur < MIN_TRADE_EUR:
        return 0.0, 0.0
    params = {}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    order = ex.create_order(pair, "market", "buy", None, None, {"cost": float(f"{eur:.2f}"), **params})
    filled = float(order.get("info", {}).get("filledAmount") or 0) or float(order.get("filled") or 0)
    avg = float(order.get("average") or order.get("price") or 0.0)
    return filled, avg

def sell_market(ex, pair: str, qty: float) -> Tuple[float, float]:
    if qty <= 0:
        return 0.0, 0.0
    params = {}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    order = ex.create_order(pair, "market", "sell", qty, None, params)
    proceeds = float(order.get("cost") or 0.0)
    avg = float(order.get("average") or order.get("price") or 0.0)
    return proceeds, avg

# ---------- Grid-logica ----------
def try_grid_live(ex, pair: str, price_now: float, price_prev: float, grid: dict, port: dict) -> List[str]:
    logs = []
    levels = grid["levels"]
    cash_alloc = port["coins"][pair]["cash_alloc"]
    ticket_eur = euro_per_ticket(cash_alloc, len(levels))

    # BUY bij neerwaartse cross
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            fee_eur = ticket_eur * FEE_PCT
            if (port["cash_eur"] - MIN_CASH_BUFFER_EUR) < (ticket_eur + fee_eur):
                logs.append(f"[{pair}] BUY skip: onvoldoende cash.")
                continue
            qty, avgp = buy_market(ex, pair, ticket_eur)
            if qty <= 0 or avgp <= 0:
                logs.append(f"[{pair}] BUY fail.")
                continue
            port["cash_eur"] -= (ticket_eur + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": avgp})
            append_csv(TRADES_CSV,
                [now_iso(), pair, "BUY", f"{ticket_eur:.2f}", f"{avgp:.6f}", f"{qty:.8f}",
                 f"{port['cash_eur']:.2f}", ""],
                ["timestamp","pair","side","eur","price","amount","cash_eur","comment"])
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{avgp:.6f} | cash=€{port['cash_eur']:.2f}")

    # SELL bij opwaartse cross (winst)
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
            proceeds, avgp = sell_market(ex, pair, qty)
            if proceeds <= 0 or avgp <= 0:
                # push terug bij falen
                grid["inventory_lots"].insert(0, lot)
                continue
            fee_eur = proceeds * FEE_PCT
            pnl = proceeds - fee_eur - (qty * lot["buy_price"])
            port["cash_eur"] += (proceeds - fee_eur)
            port["coins"][pair]["qty"] -= qty
            port["pnl_realized"] += pnl
            append_csv(TRADES_CSV,
                [now_iso(), pair, "SELL", f"{proceeds:.2f}", f"{avgp:.6f}", f"{qty:.8f}",
                 f"{port['cash_eur']:.2f}", f"pnl={pnl:.2f}"],
                ["timestamp","pair","side","eur","price","amount","cash_eur","comment"])
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{avgp:.6f} | pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}")

    grid["last_price"] = price_now
    return logs

# ---------- Main ----------
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    weights = normalize_weights(pairs, WEIGHTS_CSV)

    ex = make_ex()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten.")

    # Bescherm bestaande balans → speelpot beperken tot nieuw vrij EUR (optioneel)
    acct = ex.fetch_balance().get("free", {})
    start_eur = float(acct.get("EUR", 0) or 0)
    if LOCK_PREEXISTING_BALANCE or PROTECT_EXISTING:
        print(f"[guard] Pre-existing EUR op account: €{start_eur:.2f} (wordt beschermd).")

    print(f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={pct(FEE_PCT)} | "
          f"pairs={pairs} | weights={weights} | factor={ORDER_SIZE_FACTOR} | short={ENABLE_SHORT}")

    state = {}
    if RESET_STATE and not WARM_START:
        try:
            STATE_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        state = {}
    else:
        state = load_json(STATE_FILE, {})

    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    # open CSV headers als ze nog niet bestaan (geen rij schrijven nu; append later)
    if not TRADES_CSV.exists():
        append_csv(TRADES_CSV, [], ["timestamp","pair","side","eur","price","amount","cash_eur","comment"])
        TRADES_CSV.unlink(missing_ok=True)  # verwijder lege rij; header komt bij eerste write
    if not EQUITY_CSV.exists():
        append_csv(EQUITY_CSV, [], ["date","total_equity_eur"])
        EQUITY_CSV.unlink(missing_ok=True)

    last_report_ts = 0.0
    last_hb_ts = 0.0

    while True:
        try:
            # Per pair prijzen en grid
            for p in pairs:
                t = ex.fetch_ticker(p)
                px = float(t["last"])
                g = state["grids"][p]
                logs = try_grid_live(ex, p, px, g["last_price"], g, state["portfolio"])
                if logs:
                    print("\n".join(logs))

            # Heartbeat
            now = time.time()
            if now - last_hb_ts >= HEARTBEAT_EVERY_SEC:
                eq = mark_to_market(ex, state, pairs)
                append_csv(EQUITY_CSV, [datetime.now(timezone.utc).date().isoformat(), f"{eq:.2f}"],
                           ["date","total_equity_eur"])
                maybe_skim(state)
                cash = state["portfolio"]["cash_eur"]
                pr   = state["portfolio"]["pnl_realized"]
                day_change = eq - CAPITAL_EUR
                if DISABLE_COLOR:
                    print(f"[HB] equity=€{eq:.2f} | cash=€{cash:.2f} | pnl_realized=€{pr:.2f} | day=€{day_change:.2f}")
                else:
                    col_s = "\033[92m" if day_change >= 0 else "\033[91m"
                    col_e = "\033[0m"
                    print(f"[HB] equity=€{eq:.2f} | cash=€{cash:.2f} | pnl_realized=€{pr:.2f} | "
                          f"day={col_s}€{day_change:.2f}{col_e}")
                last_hb_ts = now

            # 4u (of ENV) rapport
            if now - last_report_ts >= REPORT_EVERY_HOURS * 3600:
                eq   = mark_to_market(ex, state, pairs)
                cash = state["portfolio"]["cash_eur"]
                pr   = state["portfolio"]["pnl_realized"]
                day_change = eq - CAPITAL_EUR
                col_s = "" if DISABLE_COLOR else ("\033[92m" if day_change >= 0 else "\033[91m")
                col_e = "" if DISABLE_COLOR else "\033[0m"
                print(f"[SUMMARY] total_eq=€{eq:.2f} | cash=€{cash:.2f} | pnl_realized=€{pr:.2f} | "
                      f"day={col_s}€{day_change:.2f}{col_e}")
                last_report_ts = now

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
