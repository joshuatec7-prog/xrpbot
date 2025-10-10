# =========================
# live_grid.py  (Bitvavo)
# =========================
# Vereiste env-variabelen (Render/Docker):
# API_KEY, API_SECRET
# Optioneel: OPERATOR_ID  → alleen invullen als Bitvavo die voor jouw key vereist
# Overig:
# EXCHANGE=bitvavo
# COINS="BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
# CAPITAL_EUR=1100
# REINVEST_PROFITS=false
# REINVEST_THRESHOLD_EUR=100
# FEE_PCT=0.0015
# GRID_LEVELS=64
# BAND_PCT=0.08
# MIN_PROFIT_PCT=0.0005
# MIN_PROFIT_EUR=0.05
# SELL_SAFETY_PCT=0.003
# ORDER_SIZE_FACTOR=1.6
# MIN_QUOTE_EUR=5
# MIN_CASH_BUFFER_EUR=25
# BUY_FREE_EUR_MIN=100
# ALLOW_BUYS=true
# LOCK_PREEXISTING_BALANCE=false
# DATA_DIR=data
# LOG_SUMMARY_SEC=240
# SLEEP_SEC=3

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import ccxt

# ---------- ENV ----------
API_KEY = os.getenv("API_KEY", "").strip()
API_SECRET = os.getenv("API_SECRET", "").strip()

_op = os.getenv("OPERATOR_ID", "").strip()
OPERATOR_ID = _op if _op not in ("", "0", "0000000000") else None

EXCHANGE = os.getenv("EXCHANGE", "bitvavo").lower()
PAIRS = [p.strip() for p in os.getenv("COINS", "BTC/EUR,ETH/EUR").split(",") if p.strip()]

CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "1100"))
REINVEST_PROFITS = os.getenv("REINVEST_PROFITS", "false").lower() in ("1", "true", "yes")
REINVEST_THRESHOLD_EUR = float(os.getenv("REINVEST_THRESHOLD_EUR", "0"))

FEE_PCT = float(os.getenv("FEE_PCT", "0.0015"))
GRID_LEVELS = int(os.getenv("GRID_LEVELS", "64"))
BAND_PCT = float(os.getenv("BAND_PCT", "0.08"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.0005"))
MIN_PROFIT_EUR = float(os.getenv("MIN_PROFIT_EUR", "0.05"))
SELL_SAFETY_PCT = float(os.getenv("SELL_SAFETY_PCT", "0.003"))
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR", "1.6"))

MIN_QUOTE_EUR = float(os.getenv("MIN_QUOTE_EUR", "5"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
BUY_FREE_EUR_MIN = float(os.getenv("BUY_FREE_EUR_MIN", "100"))
ALLOW_BUYS = os.getenv("ALLOW_BUYS", "true").lower() in ("1", "true", "yes")
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE", "false").lower() in ("1", "true", "yes")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_JSON = DATA_DIR / "live_state.json"
TRADES_CSV = DATA_DIR / "live_trades.csv"
EQUITY_CSV = DATA_DIR / "live_equity.csv"

SLEEP_SEC = int(os.getenv("SLEEP_SEC", "3"))
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC", "240"))

COL_B = "\033[1m"; COL_C = "\033[96m"; COL_G = "\033[92m"; COL_R = "\033[91m"; COL_0 = "\033[0m"

# ---------- operator-id guard ----------
def _guard_operator_error(exc: Exception):
    msg = str(exc).lower()
    if "operatorid" in msg and "required" in msg:
        raise SystemExit(
            "Bitvavo meldt: operatorId is vereist voor deze API-key. "
            "Oplossing: gebruik een normale Bitvavo API-key zonder operator-verplichting "
            "of vul een geldige OPERATOR_ID in de omgeving."
        )

# ---------- helpers ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def append_csv(path: Path, row: List[str], header: List[str] = None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header:
            w.writerow(header)
        w.writerow(row)

def to_num(x, d=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return d

# ---------- exchange ----------
def make_exchange():
    if EXCHANGE != "bitvavo":
        raise RuntimeError("Alleen Bitvavo wordt ondersteund.")
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API_KEY en API_SECRET vereist.")

    ex = ccxt.bitvavo(
        {
            "apiKey": API_KEY,
            "secret": API_SECRET,
            "enableRateLimit": True,
            "options": {"createMarketBuyOrderRequiresPrice": False},
        }
    )

    if OPERATOR_ID:
        ex.options = {**getattr(ex, "options", {}), "operatorId": OPERATOR_ID}
        if not hasattr(ex, "headers") or not isinstance(ex.headers, dict):
            ex.headers = {}
        ex.headers["BITVAVO-ACCESS-OPERATOR-ID"] = OPERATOR_ID

    try:
        ex.load_markets()
    except ccxt.BaseError as e:
        _guard_operator_error(e)
        raise
    return ex

def fetch_mid(ex, pair: str) -> float:
    t = ex.fetch_ticker(pair)
    bid = to_num(t.get("bid"), 0.0)
    ask = to_num(t.get("ask"), 0.0)
    return (bid + ask) / 2.0 if bid and ask else to_num(t.get("last"), 0.0)

def best_bid(ex, pair: str) -> float:
    return to_num(ex.fetch_ticker(pair).get("bid"), 0.0)

def best_ask(ex, pair: str) -> float:
    return to_num(ex.fetch_ticker(pair).get("ask"), 0.0)

def amount_to_precision(ex, pair: str, a: float) -> float:
    return float(ex.amount_to_precision(pair, a))

# private-call wrappers met operator guard
def _fetch_balance(ex):
    try:
        return ex.fetch_balance()
    except ccxt.BaseError as e:
        _guard_operator_error(e)
        raise

def _create_order(ex, *args, **kwargs):
    try:
        return ex.create_order(*args, **kwargs)
    except ccxt.BaseError as e:
        _guard_operator_error(e)
        raise

def free_eur(ex) -> float:
    bal = _fetch_balance(ex)
    return to_num(bal.get("EUR", {}).get("free"), 0.0)

def free_base(ex, base: str) -> float:
    bal = _fetch_balance(ex)
    return to_num(bal.get(base, {}).get("free"), 0.0)

def market_mins(ex, pair: str):
    m = ex.markets.get(pair, {})
    min_base = to_num(m.get("limits", {}).get("amount", {}).get("min"), 0.0) or 0.0
    min_quote = max(MIN_QUOTE_EUR, to_num(m.get("limits", {}).get("cost", {}).get("min"), 0.0) or 0.0)
    return (min_quote, min_base)

# ---------- grid / targets ----------
def build_levels(px_now: float) -> List[float]:
    if px_now <= 0:
        return []
    span = px_now * BAND_PCT
    step = span / GRID_LEVELS
    return [px_now - i * step for i in range(1, GRID_LEVELS + 1)]

def target_sell(buy_px: float, qty: float) -> float:
    pct_target = buy_px * (1 + 2 * FEE_PCT + MIN_PROFIT_PCT + SELL_SAFETY_PCT)
    abs_target = buy_px + (MIN_PROFIT_EUR / max(qty, 1e-12)) * (1 + FEE_PCT)
    return max(pct_target, abs_target)

def cap_now(port: Dict) -> float:
    pnl = to_num(port.get("pnl_realized", 0.0))
    if not REINVEST_PROFITS:
        return CAPITAL_EUR
    extra = pnl if pnl >= REINVEST_THRESHOLD_EUR else 0.0
    return CAPITAL_EUR + extra

def ticket_eur_for_pair(port, pair: str) -> float:
    alloc = cap_now(port) / max(len(PAIRS), 1)
    base = alloc / GRID_LEVELS
    return base * ORDER_SIZE_FACTOR

# ---------- state ----------
def init_state(ex):
    state = {
        "portfolio": {"pnl_realized": 0.0, "coins": {p: {"qty": 0.0} for p in PAIRS}},
        "pairs": {p: {"last_price": 0.0, "levels": [], "inventory_lots": []} for p in PAIRS},
        "baseline": {},
    }
    try:
        if STATE_JSON.exists():
            state.update(json.loads(STATE_JSON.read_text(encoding="utf-8")))
    except Exception:
        pass

    if LOCK_PREEXISTING_BALANCE and not state["baseline"]:
        bal = _fetch_balance(ex)
        for p in PAIRS:
            base = p.split("/")[0]
            state["baseline"][base] = to_num(bal.get(base, {}).get("free"), 0.0)
    return state

def save_state(state):
    tmp = STATE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_JSON)

def write_equity(ex, state):
    cash = free_eur(ex)
    value = 0.0
    for p in PAIRS:
        qty = to_num(state["portfolio"]["coins"][p]["qty"], 0.0)
        if qty > 0:
            value += qty * fetch_mid(ex, p)
    eq = cash + value
    append_csv(EQUITY_CSV, [now_iso(), f"{eq:.2f}"], header=["timestamp", "equity_eur"])

# ---------- orders ----------
def buy_market(ex, pair: str, cost_eur: float):
    ask = best_ask(ex, pair)
    if ask <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    qty = amount_to_precision(ex, pair, cost_eur / ask)
    if qty <= 0:
        return (0.0, 0.0, 0.0, 0.0)
    o = _create_order(ex, pair, "market", "buy", qty)
    avg = to_num(o.get("average"), ask)
    filled = to_num(o.get("filled"), qty)
    cost = avg * filled
    fee = cost * FEE_PCT
    return (filled, avg, fee, cost)

def sell_market(ex, pair: str, qty: float):
    bid = best_bid(ex, pair)
    if bid <= 0 or qty <= 0:
        return (0.0, 0.0, 0.0)
    o = _create_order(ex, pair, "market", "sell", qty)
    avg = to_num(o.get("average"), bid)
    proceeds = avg * to_num(o.get("filled"), qty)
    fee = proceeds * FEE_PCT
    return (proceeds, avg, fee)

# ---------- per pair ----------
def process_pair(ex, pair: str, state, out: List[str]):
    port = state["portfolio"]
    ps = state["pairs"][pair]

    px = fetch_mid(ex, pair)
    if px <= 0:
        out.append(f"[{pair}] geen prijs, skip.")
        return

    if not ps["levels"]:
        ps["levels"] = build_levels(px)
    if ps["last_price"] <= 0:
        ps["last_price"] = px

    # SELL
    if ps["inventory_lots"]:
        min_quote, min_base = market_mins(ex, pair)
        base = pair.split("/")[0]
        allowed = None
        if LOCK_PREEXISTING_BALANCE:
            allowed = max(0.0, free_base(ex, base) - to_num(state["baseline"].get(base, 0.0)))

        changed = True
        while changed and ps["inventory_lots"]:
            changed = False
            lot = ps["inventory_lots"][0]
            trg = target_sell(lot["buy_price"], lot["qty"])
            bid = best_bid(ex, pair)
            if bid + 1e-12 < trg:
                out.append(f"[{pair}] SELL wait: bid €{bid:.2f} < trigger €{trg:.2f} (buy €{lot['buy_price']:.2f})")
                break

            qty = lot["qty"]
            if qty < min_base or (qty * px) < min_quote:
                out.append(f"[{pair}] SELL skip: lot te klein (amt {qty:.8f} / min {min_base}, €{qty*px:.2f} / min €{min_quote:.2f})")
                break
            if allowed is not None and allowed + 1e-12 < qty:
                out.append(f"[{pair}] SELL stop: baseline-protect ({allowed:.8f} {base} vrij).")
                break

            proceeds, avg, fee = sell_market(ex, pair, amount_to_precision(ex, pair, qty))
            if proceeds <= 0 or avg <= 0:
                out.append(f"[{pair}] SELL fail: geen fill.")
                break

            pnl = proceeds - fee - (qty * lot["buy_price"] * (1 + FEE_PCT))
            port["pnl_realized"] = to_num(port.get("pnl_realized", 0.0)) + pnl
            port["coins"][pair]["qty"] = to_num(port["coins"][pair]["qty"]) - qty
            ps["inventory_lots"].pop(0)
            if allowed is not None:
                allowed -= qty

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "SELL", f"{avg:.6f}", f"{qty:.8f}", f"{proceeds:.2f}", "", pair.split("/")[0],
                 f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "take_profit"],
                header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"]
            )
            col = COL_G if pnl >= 0 else COL_R
            out.append(f"{col}[{pair}] SELL {qty:.8f} @ €{avg:.6f} | pnl=€{pnl:.2f} | trigger €{trg:.2f}{COL_0}")
            changed = True

    # BUY
    if not ALLOW_BUYS:
        out.append(f"[{pair}] BUY paused (ALLOW_BUYS=false).")
    else:
        crossed = [L for L in ps["levels"] if px < L <= ps["last_price"]]
        if crossed:
            freeE = free_eur(ex)
            invested_cost = sum(
                l["qty"] * l["buy_price"]
                for p2 in PAIRS
                for l in state["pairs"][p2]["inventory_lots"]
            )
            room = cap_now(port) - invested_cost

            for _L in crossed:
                min_quote, min_base = market_mins(ex, pair)

                if freeE < (BUY_FREE_EUR_MIN + MIN_CASH_BUFFER_EUR):
                    out.append(f"[{pair}] BUY skip: policy free_EUR<€{BUY_FREE_EUR_MIN:.2f} (buffer €{MIN_CASH_BUFFER_EUR:.2f}).")
                    continue
                if room <= 0:
                    out.append(f"[{pair}] BUY skip: cap bereikt (cap=€{cap_now(port):.2f}).")
                    continue

                ticket = ticket_eur_for_pair(port, pair)
                cost = min(ticket, max(0.0, freeE - MIN_CASH_BUFFER_EUR), room)
                cost = max(cost, min_quote)
                if cost <= 0:
                    need = max(min_quote + MIN_CASH_BUFFER_EUR - freeE, 0.0)
                    out.append(f"[{pair}] BUY skip: vrije EUR te laag (free≈€{freeE:.2f}, nodig≥€{min_quote+MIN_CASH_BUFFER_EUR:.2f}, tekort≈€{need:.2f}).")
                    continue

                qty, avg, fee, executed = buy_market(ex, pair, cost)
                if qty <= 0 or avg <= 0:
                    out.append(f"[{pair}] BUY fail: geen fill.")
                    continue
                if qty < min_base or executed < min_quote:
                    out.append(f"[{pair}] BUY fill < minima (amt={qty:.8f}/{min_base}, €{executed:.2f}/€{min_quote:.2f}).")
                    continue

                ps["inventory_lots"].append({"qty": qty, "buy_price": avg})
                port["coins"][pair]["qty"] = to_num(port["coins"][pair]["qty"]) + qty

                tgt = target_sell(avg, qty)
                append_csv(
                    TRADES_CSV,
                    [now_iso(), pair, "BUY", f"{avg:.6f}", f"{qty:.8f}", f"{executed:.2f}", "", pair.split("/")[0],
                     f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
                    header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"]
                )
                out.append(f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avg:.6f} | exec≈€{executed:.2f} → target SELL≈€{tgt:.2f}{COL_0}")

    ps["last_price"] = px

# ---------- main ----------
def main():
    ex = make_exchange()
    state = init_state(ex)
    last_summary = 0.0

    print(
        f"{COL_B}== LIVE GRID start =={COL_0} | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} "
        f"| fee={FEE_PCT*100:.3f}% | pairs={PAIRS} | buys={'ON' if ALLOW_BUYS else 'OFF'} "
        f"| buffer=€{MIN_CASH_BUFFER_EUR:.2f} | keep_free=€{BUY_FREE_EUR_MIN:.2f}"
    )

    while True:
        out: List[str] = []
        try:
            for p in PAIRS:
                process_pair(ex, p, state, out)

            if (time.time() - last_summary) >= LOG_SUMMARY_SEC:
                cash = free_eur(ex)
                live_inv = 0.0
                invested_cost = 0.0
                for p in PAIRS:
                    mid = fetch_mid(ex, p)
                    qty = to_num(state["portfolio"]["coins"][p]["qty"], 0.0)
                    live_inv += qty * mid
                    for l in state["pairs"][p]["inventory_lots"]:
                        invested_cost += l["qty"] * l["buy_price"]
                cap = cap_now(state["portfolio"])
                pnl = to_num(state["portfolio"]["pnl_realized"], 0.0)
                print(
                    f"[SUMMARY] total_eq=€{cash+live_inv:.2f} | cash=€{cash:.2f} | free_EUR=€{cash:.2f} "
                    f"| invested_cost=€{invested_cost:.2f} | live_inv=€{live_inv:.2f} "
                    f"| cap_now=€{cap:.2f} | pnl_realized=€{pnl:.2f} | buys={'ON' if ALLOW_BUYS else 'OFF'}"
                )
                write_equity(ex, state)
                save_state(state)
                last_summary = time.time()

            for line in out:
                print(line)

            time.sleep(SLEEP_SEC)

        except ccxt.BaseError as e:
            print(f"[exchange] {e}")
            time.sleep(2)
        except KeyboardInterrupt:
            print("Stop door gebruiker.")
            save_state(state)
            break
        except Exception as e:
            print(f"[fatal] {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
