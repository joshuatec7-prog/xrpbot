# live_grid.py — Multi-coin LIVE Grid Bot (Bitvavo Spot)
# - Zelfde opzet als je papergrid: band (P10–P90 fallback ±BAND_PCT), GRID_LEVELS, geometrische levels
# - Per neerwaartse cross: BUY (market, cost in euro's)
# - Per opwaartse cross t.o.v. koop-level: SELL (market amount)
# - Houdt per pair "inventory_lots" (qty + buy_price) in state.json
# - Echte fondsen: EUR en BASE-balansen via exchange
# - Veiligheid: min order-cost, buffer, rate-limit, retries
# - SHORT is uit op Bitvavo Spot (ENABLE_SHORT wordt genegeerd tenzij je later futures gebruikt)

import os, json, time, csv, random
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ======== ENV ========
LIVE = os.getenv("LIVE","true").lower() in ("1","true","yes")
if not LIVE:
    raise SystemExit("Zet LIVE=true voor live handelen.")

API_KEY    = os.getenv("API_KEY","").strip()
API_SECRET = os.getenv("API_SECRET","").strip()
if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY en/of API_SECRET ontbreekt.")

COINS_CSV = os.getenv("COINS","BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR")
WEIGHTS_CSV = os.getenv("WEIGHTS","BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10")

GRID_LEVELS = int(os.getenv("GRID_LEVELS","48"))
BAND_PCT    = float(os.getenv("BAND_PCT","0.25"))
FEE_PCT     = float(os.getenv("FEE_PCT","0.0015"))
SLEEP_SEC   = float(os.getenv("SLEEP_SEC","10"))
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC","600"))

ORDER_SIZE_FACTOR   = float(os.getenv("ORDER_SIZE_FACTOR","1.0"))
MIN_TRADE_EUR       = float(os.getenv("MIN_TRADE_EUR","5"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR","250"))

REQUEST_INTERVAL_SEC = float(os.getenv("REQUEST_INTERVAL_SEC","1.0"))
MAX_RETRIES          = int(os.getenv("MAX_RETRIES","5"))
BACKOFF_BASE         = float(os.getenv("BACKOFF_BASE","1.5"))

ENABLE_SHORT = os.getenv("ENABLE_SHORT","false").lower() in ("1","true","yes")  # spot: genegeerd
MAX_SHORT_EXPOSURE_FRAC = float(os.getenv("MAX_SHORT_EXPOSURE_FRAC","0.30"))
SHORT_ORDER_FACTOR  = float(os.getenv("SHORT_ORDER_FACTOR","1.0"))

DATA_DIR = Path(os.getenv("DATA_DIR","/var/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE  = DATA_DIR / "state.json"
TRADES_CSV  = DATA_DIR / "trades.csv"
EQUITY_CSV  = DATA_DIR / "equity.csv"

# ======== Helpers ========
def now_iso():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def append_csv(path: Path, row: List):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            if path == TRADES_CSV:
                w.writerow(["timestamp","symbol","side","price","amount","reason","pnl_eur"])
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

# ======== Exchange ========
ex = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True
})
ex.options["createMarketBuyOrderRequiresPrice"] = False
MARKETS = ex.load_markets()

def amount_to_precision(sym, amt: float) -> float:
    return float(ex.amount_to_precision(sym, amt))

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
    for i in range(1, MAX_RETRIES+1):
        try:
            respect_interval()
            return fn(*a, **kw)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            print(f"[net] retry {i}/{MAX_RETRIES}: {e}")
            time.sleep(delay + random.random()*0.5)
            delay *= BACKOFF_BASE
        except Exception as e:
            print(f"[unexpected] {e}")
            time.sleep(1.0)
    raise RuntimeError("Max retries reached")

# ======== Grid ========
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(pair: str) -> Tuple[float, float]:
    try:
        ohlcv = with_retry(ex.fetch_ohlcv, pair, timeframe="1h", limit=24*30)
        if ohlcv and len(ohlcv) >= 50:
            closes = [c[4] for c in ohlcv if c and c[4] is not None]
            s = pd.Series(closes)
            p10 = float(s.quantile(0.10))
            p90 = float(s.quantile(0.90))
            if p90 > p10 > 0:
                return p10, p90
    except Exception:
        pass
    last = float(with_retry(ex.fetch_ticker, pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def mk_grid_state(pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(pair)
    return {
        "pair": pair,
        "low": low,
        "high": high,
        "levels": geometric_levels(low, high, levels),
        "last_price": None,
        "inventory_lots": []  # [{qty, buy_price}]
    }

# ======== Sizing / Balansen ========
def euro_balance() -> float:
    bal = with_retry(ex.fetch_balance)
    free = bal.get("free", {})
    return float(free.get("EUR", 0) or 0)

def base_balance(symbol: str) -> float:
    base, _ = symbol.split("/")
    bal = with_retry(ex.fetch_balance)
    free = bal.get("free", {})
    return float(free.get(base, 0) or 0)

def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    # ~90% van allocatie over (n_levels/2) stappen; vermenigvuldigd met factor
    if n_levels < 2: n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

# ======== Orders ========
def market_buy_cost(pair: str, px: float, cost_eur: float) -> float:
    cost_eur = float(f"{cost_eur:.2f}")
    if cost_eur < MIN_TRADE_EUR:
        return 0.0
    params = {"cost": cost_eur}
    order = with_retry(ex.create_order, pair, "market", "buy", None, None, params)
    # Bitvavo geeft filledAmount in info
    filled = None
    try:
        filled = float(order.get("info", {}).get("filledAmount"))
    except Exception:
        pass
    if not filled:
        filled = cost_eur / px
    return filled

def market_sell_amount(pair: str, qty: float, px: float) -> float:
    qty = amount_to_precision(pair, qty)
    if qty * px < MIN_TRADE_EUR:
        return 0.0
    with_retry(ex.create_order, pair, "market", "sell", qty)
    return qty

# ======== P&L helper ========
def estimate_pnl_sell(qty: float, sell_px: float, buy_px: float) -> float:
    gross = qty * (sell_px - buy_px)
    fees  = (qty*sell_px + qty*buy_px) * FEE_PCT
    return gross - fees

# ======== MAIN ========
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    pairs = [p for p in pairs if p in MARKETS]
    if not pairs:
        raise SystemExit("Geen geldige COINS op exchange.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)

    # init state
    state = load_state()
    if "grids" not in state: state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(p, GRID_LEVELS)
    save_state(state)

    last_sum = 0.0
    last_equity_day = None

    print(f"== LIVE GRID start | pairs={pairs} | weights={weights} | levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% | buffer=€{MIN_CASH_BUFFER_EUR:.0f}")

    while True:
        try:
            # dagelijks equity snapshot
            today = datetime.now(timezone.utc).date().isoformat()
            if today != last_equity_day:
                # eenvoudige mark-to-market: EUR + waarde van BASE per pair
                total = euro_balance()
                for p in pairs:
                    qty = base_balance(p)
                    if qty > 0:
                        px = float(with_retry(ex.fetch_ticker, p)["last"])
                        total += qty * px
                append_csv(EQUITY_CSV, [today, f"{total:.2f}"])
                last_equity_day = today

            # beschikbare EUR & allocaties
            total_eur = euro_balance()
            alloc = {p: total_eur * weights[p] for p in pairs}

            for p in pairs:
                t = with_retry(ex.fetch_ticker, p)
                px = float(t["last"])
                grid = state["grids"][p]
                prev_px = grid["last_price"]
                levels = grid["levels"]

                # koop/verkoop tickets (in EUR)
                ticket_eur = euro_per_ticket(alloc[p], len(levels))

                # === BUY (neerwaartse cross) ===
                if prev_px is not None and px < prev_px:
                    crossed = [L for L in levels if px <= L < prev_px]
                    for L in crossed:
                        eur = euro_balance()
                        if eur - MIN_CASH_BUFFER_EUR < ticket_eur:
                            print(f"[{p}] BUY skip: onvoldoende EUR (buffer).")
                            continue
                        if ticket_eur < MIN_TRADE_EUR:
                            print(f"[{p}] BUY skip: onder min €{MIN_TRADE_EUR:.2f}.")
                            continue

                        filled = market_buy_cost(p, L, ticket_eur)
                        if filled > 0:
                            # boek lot
                            grid["inventory_lots"].append({"qty": filled, "buy_price": L})
                            append_csv(TRADES_CSV, [now_iso(), p, "BUY", f"{L:.6f}", f"{filled:.8f}", "grid_buy", f"{0.0:.2f}"])
                            print(f"[{p}] BUY {filled:.8f} @ €{L:.6f} | EUR_cash~{euro_balance():.2f}")

                # === SELL (opwaartse cross) ===
                if prev_px is not None and px > prev_px and grid["inventory_lots"]:
                    crossed = [L for L in levels if prev_px < L <= px]
                    for L in crossed:
                        # pak eerste winnende lot
                        lot_idx = None
                        for i, lot in enumerate(grid["inventory_lots"]):
                            if L > lot["buy_price"]:
                                lot_idx = i
                                break
                        if lot_idx is None:
                            continue
                        lot = grid["inventory_lots"].pop(lot_idx)
                        qty = lot["qty"]
                        sold = market_sell_amount(p, qty, L)
                        if sold > 0:
                            pnl = estimate_pnl_sell(sold, L, lot["buy_price"])
                            append_csv(TRADES_CSV, [now_iso(), p, "SELL", f"{L:.6f}", f"{sold:.8f}", "grid_sell", f"{pnl:.2f}"])
                            print(f"[{p}] SELL {sold:.8f} @ €{L:.6f} | pnl≈€{pnl:.2f} | EUR_cash~{euro_balance():.2f}")
                        else:
                            # sell niet gelukt → lot terug
                            grid["inventory_lots"].insert(0, lot)

                grid["last_price"] = px

            # periodieke samenvatting
            nowt = time.time()
            if nowt - last_sum >= LOG_SUMMARY_SEC:
                eq = euro_balance()
                for p in pairs:
                    qty = base_balance(p)
                    if qty > 0:
                        px = float(with_retry(ex.fetch_ticker, p)["last"])
                        eq += qty * px
                print(f"[SUMMARY] equity≈€{eq:.2f} | EUR_cash≈€{euro_balance():.2f}")
                last_sum = nowt

            save_state(state)
            time.sleep(SLEEP_SEC)

        except KeyboardInterrupt:
            print("Gestopt.")
            break
        except Exception as e:
            print(f"[runtime] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
