# live_grid.py — Multi-coin LIVE Grid Trader (Bitvavo, long-only)
# - Centraal EUR-budget (CAPITAL_EUR) en bescherming bestaande balances
# - P10–P90 band (30d/1h) per pair, fallback median ± BAND_PCT
# - Market orders + fee-schatting + min ticket checks
# - Optionele OPERATOR_ID: alleen meesturen als ingevuld
# - Periodieke summary + uitgebreid rapport om de X uur
# - Eenvoudige state in /var/data/live_state.json

import os, json, time, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import ccxt
import pandas as pd

# ========= ENV =========
API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY en/of API_SECRET ontbreekt.")

CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "1000"))
COINS_CSV   = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
WEIGHTS_CSV = os.getenv("WEIGHTS", "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()

FEE_PCT      = float(os.getenv("FEE_PCT", "0.0015"))
GRID_LEVELS  = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT     = float(os.getenv("BAND_PCT", "0.25"))
SLEEP_SEC    = float(os.getenv("SLEEP_SEC", "10"))
REQUEST_INTERVAL_SEC = float(os.getenv("REQUEST_INTERVAL_SEC", "1.0"))
MAX_RETRIES  = int(os.getenv("MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.7"))

MIN_TRADE_EUR       = float(os.getenv("MIN_TRADE_EUR", "5"))
ORDER_SIZE_FACTOR   = float(os.getenv("ORDER_SIZE_FACTOR", "1.0"))
MAX_SHORT_EXPO_FRAC = float(os.getenv("MAX_SHORT_EXPOSURE_FRAC", "0.0"))  # niet gebruikt (long-only)

LOCK_PREEXISTING_BAL = os.getenv("LOCK_PREEXISTING_BALANCE", "true").lower() in ("1","true","yes")

LOG_SUMMARY_SEC     = int(os.getenv("LOG_SUMMARY_SEC", "600"))
REPORT_EVERY_HOURS  = float(os.getenv("REPORT_EVERY_HOURS", "4"))
REPORT_EVERY_SEC    = int(REPORT_EVERY_HOURS * 3600)

DATA_DIR   = Path(os.getenv("DATA_DIR", "/var/data"))
STATE_FILE = DATA_DIR / os.getenv("STATE_FILE", "live_state.json")

OPERATOR_ID = os.getenv("OPERATOR_ID", "").strip()  # optioneel

# ========= Helpers =========
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def append_csv(path: Path, row: List[str], header: List[str]):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
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

# ========= Exchange =========
def make_ex():
    ex = ccxt.bitvavo({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })
    ex.options["createMarketBuyOrderRequiresPrice"] = False
    ex.load_markets()
    return ex

# ========= Grid tools =========
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    """P10–P90 van 30d 1h closes; fallback ± BAND_PCT rond last."""
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
    t = ex.fetch_ticker(pair)
    last = float(t["last"])
    return last * (1 - BAND_PCT), last * (1 + BAND_PCT)

def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    if n_levels < 2:
        n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

# ========= Robust calls =========
_last_call = 0.0
def respect_interval():
    global _last_call
    now = time.time()
    wait = REQUEST_INTERVAL_SEC - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()

def with_retry(fn, *a, **kw):
    delay = 1.0
    for i in range(1, MAX_RETRIES + 1):
        try:
            respect_interval()
            return fn(*a, **kw)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            print(f"[retry {i}/{MAX_RETRIES}] {e}")
            time.sleep(delay + random.random() * 0.5)
            delay *= BACKOFF_BASE
        except Exception as e:
            print(f"[unexpected] {e}")
            time.sleep(1.0)
    raise RuntimeError("Max retries reached.")

# ========= State =========
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADES_CSV = DATA_DIR / "live_trades.csv"
EQUITY_CSV = DATA_DIR / "live_equity.csv"

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_state(st: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

# ========= Orders =========
def market_buy_cost(ex, pair: str, eur_cost: float) -> float:
    """Koop met 'cost' in EUR. Retourneer approx filled base qty."""
    eur_cost = float(f"{eur_cost:.2f}")  # Bitvavo cost 2 decimals
    params = {"cost": eur_cost}
    if OPERATOR_ID:
        # Alleen meesturen als je er eentje hebt
        params["operatorId"] = OPERATOR_ID
    order = with_retry(ex.create_order, pair, "market", "buy", None, None, params)
    # fallback qty schatting:
    px = float(with_retry(ex.fetch_ticker, pair)["last"])
    try:
        filled = float(order.get("info", {}).get("filledAmount"))
        if not filled:
            filled = eur_cost / px
    except Exception:
        filled = eur_cost / px
    return float(filled)

def market_sell_amount(ex, pair: str, qty: float) -> float:
    params = {}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    with_retry(ex.create_order, pair, "market", "sell", qty, None, params)
    return qty

# ========= Main =========
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    ex = make_ex()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden op Bitvavo.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)

    # Startbalans voor bescherming
    bal0 = with_retry(ex.fetch_balance).get("free", {})
    start_eur = float(bal0.get("EUR", 0) or 0)

    # Init state
    st = load_state()
    if "portfolio" not in st:
        st["portfolio"] = {
            "budget_eur": CAPITAL_EUR,          # max budget
            "pnl_realized": 0.0,
            "coins": {p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs},
        }
    if "grids" not in st:
        st["grids"] = {}

    # Maak grids aan
    for p in pairs:
        if p not in st["grids"]:
            lo, hi = compute_band_from_history(ex, p)
            st["grids"][p] = {
                "pair": p,
                "low": lo,
                "high": hi,
                "levels": geometric_levels(lo, hi, GRID_LEVELS),
                "last_px": None,
                "lots": [],   # [{qty, buy_px}]
            }

    save_state(st)

    last_sum = 0.0
    last_report = 0.0
    print(f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% | pairs={pairs} | weights={weights} | lock_preexisting={LOCK_PREEXISTING_BAL}")

    while True:
        try:
            # Dag equity logging (per dag 1 regel)
            today = datetime.now(timezone.utc).date().isoformat()
            eq = calc_equity(ex, st, pairs)
            _append_equity_if_new_day(today, eq)

            # Per pair: ophalen prijs en gridfills
            for p in pairs:
                t = with_retry(ex.fetch_ticker, p)
                px = float(t["last"])
                g  = st["grids"][p]
                logs = try_fill_live(ex, p, px, g, st, start_eur)
                if logs:
                    print("\n".join(logs))

            now = time.time()
            if now - last_sum >= LOG_SUMMARY_SEC:
                cash = avail_eur_for_bot(ex, start_eur, st["portfolio"]["budget_eur"])
                print(f"[SUMMARY] equity=€{calc_equity(ex, st, pairs):.2f} | cash=€{cash:.2f} | pnl_realized=€{st['portfolio']['pnl_realized']:.2f}")
                last_sum = now

            if now - last_report >= REPORT_EVERY_SEC:
                run_report(ex, st, pairs)
                last_report = now

            save_state(st)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[neterr] {e}; backoff…")
            time.sleep(2 + random.random())
        except Exception as e:
            print(f"[runtime] {e}")
            time.sleep(5)

# ========= Portfolio helpers =========
def avail_eur_for_bot(ex, start_eur: float, budget: float) -> float:
    bal = with_retry(ex.fetch_balance).get("free", {})
    eur = float(bal.get("EUR", 0) or 0)
    if LOCK_PREEXISTING_BAL:
        # alleen EUR boven start_eur gebruiken, gecapt op budget
        return max(0.0, min(budget, eur - start_eur))
    # anders gewoon wat er is, gecapt op budget
    return max(0.0, min(budget, eur))

def calc_equity(ex, st: dict, pairs: List[str]) -> float:
    total = avail_eur_for_bot(ex, 0.0, 10**12)  # totale vrij EUR, maar we waarderen holdings ook
    total = 0.0
    # cash = globale EUR in wallet (alles)
    bal = with_retry(ex.fetch_balance).get("free", {})
    total += float(bal.get("EUR", 0) or 0)
    for p in pairs:
        qty = st["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(with_retry(ex.fetch_ticker, p)["last"])
            total += qty * px
    return total

def _append_equity_if_new_day(date_str: str, eq: float):
    append_csv(EQUITY_CSV, [date_str, f"{eq:.2f}"], ["date","total_equity_eur"])

# ========= Fill logic =========
def try_fill_live(ex, pair: str, px_now: float, grid: dict, port: dict, start_eur: float) -> List[str]:
    logs = []
    levels = grid["levels"]
    px_prev = grid["last_px"]

    # Budget per pair
    cash_alloc = port["portfolio"]["coins"][pair]["cash_alloc"]
    ticket_eur = euro_per_ticket(cash_alloc, len(levels))

    # BUY bij neerwaartse cross
    if px_prev is not None and px_now < px_prev:
        crossed = [L for L in levels if px_now <= L < px_prev]
        for L in crossed:
            # Is er vrije EUR beschikbaar?
            free_eur = avail_eur_for_bot(ex, start_eur, port["portfolio"]["budget_eur"])
            fee_eur  = ticket_eur * FEE_PCT
            if free_eur < (ticket_eur + fee_eur):
                logs.append(f"[{pair}] BUY skip: onvoldoende EUR (free≈€{free_eur:.2f}).")
                continue
            if ticket_eur < MIN_TRADE_EUR:
                logs.append(f"[{pair}] BUY skip: onder min €{MIN_TRADE_EUR:.2f}.")
                continue

            qty = market_buy_cost(ex, pair, ticket_eur)
            # state bijwerken
            port["portfolio"]["coins"][pair]["qty"] += qty
            grid["lots"].append({"qty": qty, "buy_px": L})

            append_csv(TRADES_CSV,
                [now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}", f"{ticket_eur*FEE_PCT:.2f}"],
                ["timestamp","pair","side","price","amount","fee_eur"]
            )
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{L:.6f}")

    # SELL bij opwaartse cross (neem eerst beste winning-lot)
    if px_prev is not None and px_now > px_prev and grid["lots"]:
        crossed = [L for L in levels if px_prev < L <= px_now]
        for L in crossed:
            idx = None
            for i, lot in enumerate(grid["lots"]):
                if L > lot["buy_px"]:
                    idx = i
                    break
            if idx is None:
                continue

            lot = grid["lots"].pop(idx)
            qty = lot["qty"]
            proceeds = qty * L
            fee_eur  = proceeds * FEE_PCT
            pnl      = proceeds - fee_eur - (qty * lot["buy_px"])

            market_sell_amount(ex, pair, qty)
            port["portfolio"]["coins"][pair]["qty"] -= qty
            port["portfolio"]["pnl_realized"] += pnl

            append_csv(TRADES_CSV,
                [now_iso(), pair, "SELL", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}"],
                ["timestamp","pair","side","price","amount","fee_eur"]
            )
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{L:.6f} | pnl=€{pnl:.2f}")

    grid["last_px"] = px_now
    return logs

# ========= Report =========
def run_report(ex, st: dict, pairs: List[str]):
    print("\n-- Rapport --")
    eq = calc_equity(ex, st, pairs)
    cash = avail_eur_for_bot(ex, 0.0, 10**12)
    pr   = st["portfolio"]["pnl_realized"]
    print(f"Equity totaal: €{eq:.2f} | Cash: €{cash:.2f} | Realized: €{pr:.2f}")
    for p in pairs:
        qty = st["portfolio"]["coins"][p]["qty"]
        print(f"{p:7s} qty={qty:.8f}")

if __name__ == "__main__":
    main()
