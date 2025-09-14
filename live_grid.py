# live_grid.py — Multi-coin LIVE Grid Bot (Bitvavo, spot)
# - Meerdere pairs (COINS)
# - Startkapitaal CAPTIAL_EUR (default €1000) => winst wordt afgeroomd (niet herbelegd)
# - Verkoopt NOOIT jouw pre-existente holdings (alleen bot-inventory)
# - State + trades/equity logs in ./data (Render-friendly)
# - GEEN echte shorts op live Bitvavo (ENABLE_SHORT moet False blijven)

import os, json, time, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# .env loader (werkt zowel lokaal als op Render)
try:
    from dotenv import load_dotenv
    load_dotenv(".env.live")
except Exception:
    pass

import ccxt
import pandas as pd

# ============ ENV ============
API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY / API_SECRET ontbreekt. Zet ze in .env.live of in Render > Environment.")

COINS_CSV = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").upper()
COINS     = [x.strip() for x in COINS_CSV.split(",") if x.strip()]

CAPITAL_EUR            = float(os.getenv("CAPITAL_EUR", "1000"))   # vaste speelpot
FEE_PCT                = float(os.getenv("FEE_PCT", "0.0015"))     # 0.15%
GRID_LEVELS            = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT               = float(os.getenv("BAND_PCT", "0.20"))
ORDER_SIZE_FACTOR      = float(os.getenv("ORDER_SIZE_FACTOR", "1.0"))
MIN_TRADE_EUR          = float(os.getenv("MIN_TRADE_EUR", "5"))
MIN_CASH_BUFFER_EUR    = float(os.getenv("MIN_CASH_BUFFER_EUR", "50"))   # hou altijd buffer aan
LOG_SUMMARY_SEC        = int(os.getenv("LOG_SUMMARY_SEC", "600"))
REPORT_EVERY_HOURS     = float(os.getenv("REPORT_EVERY_HOURS", "6"))
DATA_DIR               = Path(os.getenv("DATA_DIR", "./data"))
ENABLE_SHORT           = os.getenv("ENABLE_SHORT", "false").lower() in ("1","true","yes")  # LIVE = False
MAX_RETRIES            = int(os.getenv("MAX_RETRIES", "5"))
REQUEST_INTERVAL_SEC   = float(os.getenv("REQUEST_INTERVAL_SEC", "1.0"))
BACKOFF_BASE           = float(os.getenv("BACKOFF_BASE", "1.5"))
OPERATOR_ID            = os.getenv("OPERATOR_ID", "").strip()

# ============ Paden ============
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "live_state.json"
TRADES_CSV = DATA_DIR / "live_trades.csv"
EQUITY_CSV = DATA_DIR / "live_equity.csv"

# ============ Helpers ============
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def cprint(msg: str, color: str = ""):
    # ANSI kleur (alleen voor terminal)
    if color:
        print(f"\033[{color}m{msg}\033[0m", flush=True)
    else:
        print(msg, flush=True)

def append_csv(path: Path, row: List):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            if path == TRADES_CSV:
                w.writerow(["timestamp","pair","side","price","amount","fee_eur","cash_eur","bot_qty","realized_pnl_eur","comment"])
            elif path == EQUITY_CSV:
                w.writerow(["date","total_equity_eur","trading_equity_eur","profit_pot_eur"])
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

# ============ Exchange ============
ex = ccxt.bitvavo({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
ex.options["createMarketBuyOrderRequiresPrice"] = False
markets = ex.load_markets()
COINS = [p for p in COINS if p in markets]
if not COINS:
    raise SystemExit("Geen geldige markt in COINS gevonden op Bitvavo.")

def with_retry(fn, *a, **kw):
    delay = 1.0
    for i in range(1, MAX_RETRIES+1):
        try:
            time.sleep(max(0.0, REQUEST_INTERVAL_SEC))
            return fn(*a, **kw)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            cprint(f"[retry {i}/{MAX_RETRIES}] {e}", "33")
            time.sleep(delay + random.random()*0.5)
            delay *= BACKOFF_BASE
        except Exception as e:
            cprint(f"[unexpected] {e}", "31")
            time.sleep(1.0)
    raise RuntimeError("Max retries reached")

def amount_to_precision(sym, amt: float) -> float:
    return float(ex.amount_to_precision(sym, amt))

# ============ Grid bouw ============
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(pair: str) -> Tuple[float,float]:
    try:
        ohlcv = with_retry(ex.fetch_ohlcv, pair, timeframe="1h", limit=24*30)
        closes = [c[4] for c in ohlcv if c and c[4] is not None]
        if len(closes) >= 50:
            s = pd.Series(closes)
            p10 = float(s.quantile(0.10))
            p90 = float(s.quantile(0.90))
            if p90 > p10 > 0:
                return p10, p90
    except Exception:
        pass
    last = float(with_retry(ex.fetch_ticker, pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

# ============ Order helpers ============
def market_buy_cost(pair: str, euro_cost: float, px_hint: Optional[float] = None) -> float:
    euro_cost = float(f"{euro_cost:.2f}")
    if euro_cost < MIN_TRADE_EUR:
        return 0.0
    params = {"cost": euro_cost}
    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
    order = with_retry(ex.create_order, pair, "market", "buy", None, None, params)
    filled = None
    try:
        filled = float(order.get("info", {}).get("filledAmount"))
    except Exception:
        pass
    if not filled and px_hint:
        filled = euro_cost / px_hint
    return float(filled or 0.0)

def market_sell_qty(pair: str, qty: float) -> float:
    qty = amount_to_precision(pair, qty)
    if qty <= 0: return 0.0
    params = {}
    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
    with_retry(ex.create_order, pair, "market", "sell", qty, None, params)
    return qty

# ============ Portfolio / equity ============
def bot_euro_room(state: dict) -> float:
    """Cash dat de bot mág gebruiken (vaste pot + geen winst-herbelegging)."""
    # Centrale EUR-cash die de bot hanteert:
    cash = float(state["wallet"]["eur_cash"])
    # Trading equity = min(cash + marktwaarde bot-qty, CAPITAL_EUR). Winst gaat naar profit_pot.
    return max(0.0, cash - MIN_CASH_BUFFER_EUR)

def mark_to_market(state: dict) -> Tuple[float,float]:
    """(total_equity, trading_equity) — total = cash + bot-holdings waarde + profit_pot"""
    cash = float(state["wallet"]["eur_cash"])
    value = 0.0
    for p in COINS:
        qty = float(state["inventory"][p]["bot_qty"])
        if qty > 0:
            px = float(with_retry(ex.fetch_ticker, p)["last"])
            value += qty * px
    total_equity = cash + value + float(state["wallet"]["profit_pot_eur"])
    trading_equity = min(cash + value, CAPITAL_EUR)
    return total_equity, trading_equity

def euro_per_ticket(state: dict, pair: str, grid_levels: int) -> float:
    alloc = CAPITAL_EUR / max(1,len(COINS))  # gelijke verdeling per pair
    base = (alloc * 0.90) / max(2, grid_levels//2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

# ============ Fills (alleen bot-inventory) ============
def try_fill(pair: str, state: dict, price_now: float, price_prev: Optional[float], grid: dict) -> List[str]:
    logs = []
    levels = grid["levels"]
    order_eur = euro_per_ticket(state, pair, len(levels))

    # BUY (neerwaartse cross)
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            fee_eur = order_eur * FEE_PCT
            room = bot_euro_room(state)
            if room < (order_eur + fee_eur):
                logs.append(f"[{pair}] BUY skip: te weinig bot-cash (room={room:.2f}).")
                continue
            qty = market_buy_cost(pair, order_eur, L)
            if qty <= 0:
                logs.append(f"[{pair}] BUY fail: geen fill.")
                continue

            state["wallet"]["eur_cash"] -= (order_eur + fee_eur)
            state["inventory"][pair]["bot_qty"] += qty
            state["inventory"][pair]["lots"].append({"qty": qty, "buy_price": L})

            append_csv(TRADES_CSV, [now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}",
                                    f"{fee_eur:.2f}", f"{state['wallet']['eur_cash']:.2f}",
                                    f"{state['inventory'][pair]['bot_qty']:.8f}", f"{0.0:.2f}", "grid_buy"])
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{L:.6f} | bot_cash={state['wallet']['eur_cash']:.2f}")

    # SELL (opwaartse cross) — alleen op bot-qty (géén verkoop van jouw pre-existente holdings)
    if price_prev is not None and price_now > price_prev and state["inventory"][pair]["lots"]:
        crossed = [L for L in levels if price_prev < L <= price_now]
        for L in crossed:
            # pak eerste winstgevende lot
            idx = None
            for i, lot in enumerate(state["inventory"][pair]["lots"]):
                if L > lot["buy_price"]:
                    idx = i; break
            if idx is None:
                continue

            lot = state["inventory"][pair]["lots"].pop(idx)
            qty = lot["qty"]
            proceeds = qty * L
            fee_eur = proceeds * FEE_PCT
            pnl = proceeds - fee_eur - qty * lot["buy_price"]

            sold = market_sell_qty(pair, qty)
            if sold <= 0:
                # push lot terug als het niet lukte
                state["inventory"][pair]["lots"].insert(0, lot)
                continue

            state["inventory"][pair]["bot_qty"] -= sold
            state["wallet"]["eur_cash"] += (proceeds - fee_eur)

            # winst meteen "afromen": verplaats PnL naar profit_pot en haal uit cash
            if pnl > 0:
                state["wallet"]["eur_cash"] -= pnl
                state["wallet"]["profit_pot_eur"] += pnl

            append_csv(TRADES_CSV, [now_iso(), pair, "SELL", f"{L:.6f}", f"{sold:.8f}",
                                    f"{fee_eur:.2f}", f"{state['wallet']['eur_cash']:.2f}",
                                    f"{state['inventory'][pair]['bot_qty']:.8f}", f"{pnl:.2f}", "grid_sell"])
            logs.append(f"[{pair}] SELL {sold:.8f} @ €{L:.6f} | pnl=€{pnl:.2f} | profit_pot=€{state['wallet']['profit_pot_eur']:.2f} | cash={state['wallet']['eur_cash']:.2f}")

    grid["last_price"] = price_now
    return logs

# ============ Init state ============
def init_state() -> dict:
    st = load_state()
    if not st:
        st = {
            "wallet": {
                "eur_cash": CAPITAL_EUR,      # start pot in EUR — de bot gebruikt z'n eigen pot
                "profit_pot_eur": 0.0         # afgeroomde winst (niet meer herbelegd)
            },
            "inventory": {},
            "grids": {}
        }
    for p in COINS:
        if p not in st["inventory"]:
            st["inventory"][p] = {"bot_qty": 0.0, "lots": []}
        if p not in st["grids"]:
            low, high = compute_band_from_history(p)
            st["grids"][p] = {
                "low": low, "high": high,
                "levels": geometric_levels(low, high, GRID_LEVELS),
                "last_price": None
            }
    save_state(st)
    return st

# ============ Main loop ============
def main():
    state = init_state()
    cprint(f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% | pairs={COINS} | short={ENABLE_SHORT}", "36")

    last_sum = 0.0
    last_report = 0.0

    while True:
        try:
            # per pair prijzen ophalen + fills proberen
            for p in COINS:
                t = with_retry(ex.fetch_ticker, p)
                px = float(t["last"])
                logs = try_fill(p, state, px, state["grids"][p]["last_price"], state["grids"][p])
                for line in logs:
                    print(line, flush=True)

            # periodieke summaries
            now = time.time()
            if now - last_sum >= LOG_SUMMARY_SEC:
                total_eq, trade_eq = mark_to_market(state)
                # dag-winst (gekleurd): verschil van profit_pot vandaag — simplificatie: toon actuele pot
                cprint(f"[SUMMARY] total_eq=€{total_eq:.2f} | trading_eq=€{trade_eq:.2f} | profit_pot=€{state['wallet']['profit_pot_eur']:.2f}", "32")
                last_sum = now

            # 6-uurs rapport (één regel in equity.csv per dag, maar hier loggen we ook 6-uurs snapshot)
            if now - last_report >= REPORT_EVERY_HOURS * 3600:
                total_eq, trade_eq = mark_to_market(state)
                append_csv(EQUITY_CSV, [datetime.now(timezone.utc).date().isoformat(), f"{total_eq:.2f}", f"{trade_eq:.2f}", f"{state['wallet']['profit_pot_eur']:.2f}"])
                cprint(f"[runner] Rapport klaar. Wacht {int(REPORT_EVERY_HOURS)} uur …", "34")
                last_report = now

            save_state(state)
            time.sleep(2.0)

        except KeyboardInterrupt:
            cprint("Gestopt.", "31")
            break
        except Exception as e:
            cprint(f"[runtime] {e}", "31")
            time.sleep(5)

if __name__ == "__main__":
    if ENABLE_SHORT:
        cprint("WAARSCHUWING: ENABLE_SHORT=True is niet geschikt voor Bitvavo spot live. Zet deze op False.", "33")
    main()
