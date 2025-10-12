# =========================
# live_grid.py — Bitvavo GRID met winstbescherming
# =========================
# ENV (exacte namen)
# API_KEY, API_SECRET
# OPERATOR_ID                # optioneel; numeriek; alleen zetten als Bitvavo het vereist
# ALLOW_BUYS=true
# BAND_PCT=0.06
# BE_SAFETY_PCT=0.0005       # break-even buffer (0.05%)
# BUY_FREE_EUR_MIN=100
# CAPITAL_EUR=1100
# COINS=BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,DOGE/EUR
# DATA_DIR=data
# EXCHANGE=bitvavo
# FEE_PCT=0.0015
# GRID_LEVELS=96
# LOCK_PREEXISTING_BALANCE=false
# LOG_SUMMARY_SEC=240
# MIN_CASH_BUFFER_EUR=25
# MIN_PROFIT_EUR=0.06
# MIN_PROFIT_PCT=0.006       # 0.6%
# MIN_QUOTE_EUR=5
# ORDER_SIZE_FACTOR=2.5
# REINVEST_PROFITS=true
# REINVEST_THRESHOLD_EUR=100
# SLEEP_SEC=3
# SELL_SAFETY_PCT=0.0015
# TRAIL_PROFIT_PCT=0.004     # 0.4%

import csv, json, os, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
import ccxt

# ---------- ENV ----------
API_KEY   = os.getenv("API_KEY","").strip()
API_SECRET= os.getenv("API_SECRET","").strip()
_raw_op   = os.getenv("OPERATOR_ID","").strip()
OPERATOR_ID = int(_raw_op) if _raw_op.isdigit() and _raw_op!="0" else None

EXCHANGE  = os.getenv("EXCHANGE","bitvavo").lower()
PAIRS     = [p.strip() for p in os.getenv("COINS","BTC/EUR,ETH/EUR").split(",") if p.strip()]

CAPITAL_EUR  = float(os.getenv("CAPITAL_EUR","1100"))
REINVEST_PROFITS = os.getenv("REINVEST_PROFITS","false").lower() in ("1","true","yes")
REINVEST_THRESHOLD_EUR = float(os.getenv("REINVEST_THRESHOLD_EUR","0"))

FEE_PCT        = float(os.getenv("FEE_PCT","0.0015"))
GRID_LEVELS    = int(os.getenv("GRID_LEVELS","96"))
BAND_PCT       = float(os.getenv("BAND_PCT","0.06"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT","0.006"))
MIN_PROFIT_EUR = float(os.getenv("MIN_PROFIT_EUR","0.06"))
SELL_SAFETY_PCT= float(os.getenv("SELL_SAFETY_PCT","0.0015"))
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR","2.5"))

MIN_QUOTE_EUR       = float(os.getenv("MIN_QUOTE_EUR","5"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR","25"))
BUY_FREE_EUR_MIN    = float(os.getenv("BUY_FREE_EUR_MIN","100"))
ALLOW_BUYS          = os.getenv("ALLOW_BUYS","true").lower() in ("1","true","yes")
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE","false").lower() in ("1","true","yes")

TRAIL_PROFIT_PCT = float(os.getenv("TRAIL_PROFIT_PCT","0.004"))
BE_SAFETY_PCT    = float(os.getenv("BE_SAFETY_PCT","0.0005"))

DATA_DIR   = Path(os.getenv("DATA_DIR","data")); DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_JSON = DATA_DIR/"live_state.json"
TRADES_CSV= DATA_DIR/"live_trades.csv"
EQUITY_CSV= DATA_DIR/"live_equity.csv"

SLEEP_SEC       = int(os.getenv("SLEEP_SEC","3"))
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC","240"))

COL_B="\033[1m"; COL_C="\033[96m"; COL_G="\033[92m"; COL_R="\033[91m"; COL_0="\033[0m"

# ---------- guards ----------
def _guard_op_required(e: Exception):
    msg = str(e).lower()
    if "operatorid" in msg and "required" in msg:
        print("Bitvavo: operatorId vereist voor orders. Zet numerieke OPERATOR_ID of gebruik een normale key.", flush=True)
        sys.exit(1)

# ---------- utils ----------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def append_csv(path: Path, row: List[str], header: List[str] = None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header: w.writerow(header)
        w.writerow(row)

def to_num(x, d=0.0) -> float:
    try: return float(x)
    except: return d

# ---------- exchange ----------
def make_exchange():
    if EXCHANGE != "bitvavo": raise RuntimeError("Alleen Bitvavo.")
    if not API_KEY or not API_SECRET: raise RuntimeError("API_KEY en API_SECRET vereist.")
    ex = ccxt.bitvavo({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"createMarketBuyOrderRequiresPrice": False},
    })
    try: ex.load_markets()
    except ccxt.BaseError as e: _guard_op_required(e); raise
    return ex

def fetch_mid(ex, pair):
    t = ex.fetch_ticker(pair); b = to_num(t.get("bid"),0.0); a = to_num(t.get("ask"),0.0)
    return (b+a)/2.0 if b and a else to_num(t.get("last"),0.0)

def best_bid(ex, pair): return to_num(ex.fetch_ticker(pair).get("bid"),0.0)
def best_ask(ex, pair): return to_num(ex.fetch_ticker(pair).get("ask"),0.0)
def amount_to_precision(ex, pair, a): return float(ex.amount_to_precision(pair, a))

# alleen bij orders operatorId meesturen
def _create_order(ex, *args, **kwargs):
    params = kwargs.pop("params", {})
    if OPERATOR_ID is not None:
        params = {**params, "operatorId": OPERATOR_ID}
    try: return ex.create_order(*args, params=params, **kwargs)
    except ccxt.BaseError as e: _guard_op_required(e); raise

def _fetch_balance(ex):
    try: return ex.fetch_balance()
    except ccxt.BaseError as e: _guard_op_required(e); raise

def free_eur(ex) -> float:
    bal = _fetch_balance(ex); return to_num(bal.get("EUR",{}).get("free"), 0.0)

def free_base(ex, base) -> float:
    bal = _fetch_balance(ex); return to_num(bal.get(base,{}).get("free"), 0.0)

def market_mins(ex, pair):
    m = ex.markets.get(pair, {})
    min_base  = to_num(m.get("limits",{}).get("amount",{}).get("min"), 0.0) or 0.0
    min_quote = max(MIN_QUOTE_EUR, to_num(m.get("limits",{}).get("cost",{}).get("min"), 0.0) or 0.0)
    return (min_quote, min_base)

# ---------- grid ----------
def build_levels(px_now: float) -> List[float]:
    if px_now <= 0: return []
    span = px_now * BAND_PCT
    step = span / GRID_LEVELS
    return [px_now - i*step for i in range(1, GRID_LEVELS+1)]

def target_sell(buy_px: float, qty: float) -> float:
    pct_target = buy_px * (1 + 2*FEE_PCT + MIN_PROFIT_PCT + SELL_SAFETY_PCT)
    abs_target = buy_px + (MIN_PROFIT_EUR / max(qty,1e-12)) * (1 + FEE_PCT)
    return max(pct_target, abs_target)

def cap_now(port: Dict) -> float:
    pnl = to_num(port.get("pnl_realized",0.0))
    if not REINVEST_PROFITS: return CAPITAL_EUR
    extra = pnl if pnl >= REINVEST_THRESHOLD_EUR else 0.0
    return CAPITAL_EUR + extra

def ticket_eur_for_pair(port, pair) -> float:
    alloc = cap_now(port) / max(len(PAIRS),1)
    base  = alloc / GRID_LEVELS
    return base * ORDER_SIZE_FACTOR

# ---------- state ----------
def init_state(ex):
    state = {
        "portfolio": {"pnl_realized": 0.0, "coins": {p: {"qty": 0.0} for p in PAIRS}},
        "pairs": {p: {"last_price": 0.0, "levels": [], "inventory_lots": []} for p in PAIRS},
        "baseline": {}
    }
    try:
        if STATE_JSON.exists():
            state.update(json.loads(STATE_JSON.read_text(encoding="utf-8")))
    except: pass

    if LOCK_PREEXISTING_BALANCE and not state["baseline"]:
        bal = _fetch_balance(ex)
        for p in PAIRS:
            base = p.split("/")[0]
            state["baseline"][base] = to_num(bal.get(base,{}).get("free"),0.0)
    return state

def save_state(state):
    tmp = STATE_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_JSON)

def write_equity(ex, state):
    cash = free_eur(ex); value = 0.0
    for p in PAIRS:
        qty = to_num(state["portfolio"]["coins"][p]["qty"],0.0)
        if qty>0: value += qty * fetch_mid(ex,p)
    append_csv(EQUITY_CSV, [now_iso(), f"{cash+value:.2f}"], header=["timestamp","equity_eur"])

# ---------- orders ----------
def buy_market(ex, pair, cost_eur):
    ask = best_ask(ex, pair)
    if ask <= 0: return (0.0,0.0,0.0,0.0)
    qty = amount_to_precision(ex, pair, cost_eur/ask)
    if qty <= 0: return (0.0,0.0,0.0,0.0)
    o = _create_order(ex, pair, "market", "buy", qty)
    avg = to_num(o.get("average"), ask)
    filled = to_num(o.get("filled"), qty)
    cost = avg * filled
    fee  = cost * FEE_PCT
    return (filled, avg, fee, cost)

def sell_market(ex, pair, qty):
    bid = best_bid(ex, pair)
    if bid <= 0 or qty <= 0: return (0.0,0.0,0.0)
    o = _create_order(ex, pair, "market", "sell", qty)
    avg = to_num(o.get("average"), bid)
    proceeds = avg * to_num(o.get("filled"), qty)
    fee = proceeds * FEE_PCT
    return (proceeds, avg, fee)

# ---------- loop per pair ----------
def process_pair(ex, pair, state, out: List[str]):
    port = state["portfolio"]; ps = state["pairs"][pair]

    px = fetch_mid(ex, pair)
    if px <= 0: out.append(f"[{pair}] geen prijs, skip."); return

    if not ps["levels"]: ps["levels"] = build_levels(px)
    if ps["last_price"] <= 0: ps["last_price"] = px

    # SELL
    if ps["inventory_lots"]:
        min_quote, min_base = market_mins(ex, pair)
        base = pair.split("/")[0]
        allowed = None
        if LOCK_PREEXISTING_BALANCE:
            allowed = max(0.0, free_base(ex, base) - to_num(state["baseline"].get(base,0.0)))

        changed = True
        while changed and ps["inventory_lots"]:
            changed = False
            bid = best_bid(ex, pair)
            lot = ps["inventory_lots"][0]
            lot["peak"] = max(lot.get("peak", lot["buy_price"]), bid)
            armed = lot.get("armed", False)

            trg = target_sell(lot["buy_price"], lot["qty"])
            qty = lot["qty"]

            if qty < min_base or (qty*px) < min_quote:
                out.append(f"[{pair}] SELL skip: lot te klein"); break
            if allowed is not None and allowed + 1e-12 < qty:
                out.append(f"[{pair}] SELL stop: baseline-protect"); break

            # arming zodra winstdrempel gehaald is
            if not armed and bid >= lot["buy_price"] * (1 + MIN_PROFIT_PCT):
                lot["armed"] = True
                armed = True

            sell_reason = None
            if bid + 1e-12 >= trg:
                sell_reason = "take_profit"
            elif armed and TRAIL_PROFIT_PCT > 0 and bid <= lot["peak"] * (1 - TRAIL_PROFIT_PCT):
                sell_reason = "trailing_tp"
            elif armed and BE_SAFETY_PCT > 0 and bid <= lot["buy_price"] * (1 + BE_SAFETY_PCT):
                sell_reason = "breakeven_stop"
            else:
                out.append(f"[{pair}] SELL wait: bid €{bid:.4f} < trg €{trg:.4f} (peak €{lot['peak']:.4f})"); break

            proceeds, avg, fee = sell_market(ex, pair, amount_to_precision(ex, pair, qty))
            if proceeds <= 0 or avg <= 0:
                out.append(f"[{pair}] SELL fail: geen fill."); break

            pnl = proceeds - fee - (qty * lot["buy_price"] * (1 + FEE_PCT))
            port["pnl_realized"] = to_num(port.get("pnl_realized",0.0)) + pnl
            port["coins"][pair]["qty"] = to_num(port["coins"][pair]["qty"]) - qty
            ps["inventory_lots"].pop(0)
            if allowed is not None: allowed -= qty

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "SELL", f"{avg:.6f}", f"{qty:.8f}", f"{proceeds:.2f}", "", base,
                 f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", sell_reason],
                header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"]
            )
            col = COL_G if pnl >= 0 else COL_R
            out.append(f"{col}[{pair}] SELL {qty:.8f} @ €{avg:.6f} | pnl=€{pnl:.2f} | reason={sell_reason}{COL_0}")
            changed = True

    # BUY
    if not ALLOW_BUYS:
        out.append(f"[{pair}] BUY paused.")
    else:
        crossed = [L for L in ps["levels"] if px < L <= ps["last_price"]]
        if crossed:
            freeE = free_eur(ex)
            invested_cost = sum(l["qty"]*l["buy_price"] for p2 in PAIRS for l in state["pairs"][p2]["inventory_lots"])
            room = cap_now(port) - invested_cost

            for _ in crossed:
                min_quote, min_base = market_mins(ex, pair)

                if freeE < (BUY_FREE_EUR_MIN + MIN_CASH_BUFFER_EUR):
                    out.append(f"[{pair}] BUY skip: policy cash"); continue
                if room <= 0:
                    out.append(f"[{pair}] BUY skip: cap bereikt"); continue

                ticket = ticket_eur_for_pair(port, pair)
                cost   = min(ticket, max(0.0, freeE - MIN_CASH_BUFFER_EUR), room)
                cost   = max(cost, min_quote)
                if cost <= 0:
                    need = max(min_quote + MIN_CASH_BUFFER_EUR - freeE, 0.0)
                    out.append(f"[{pair}] BUY skip: free≈€{freeE:.2f}, nodig≥€{min_quote+MIN_CASH_BUFFER_EUR:.2f}, tekort≈€{need:.2f}"); continue

                qty, avg, fee, executed = buy_market(ex, pair, cost)
                if qty <= 0 or avg <= 0:
                    out.append(f"[{pair}] BUY fail: geen fill."); continue
                if qty < min_base or executed < min_quote:
                    out.append(f"[{pair}] BUY < minima"); continue

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
    try:
        ex = make_exchange()
    except Exception as e:
        print(f"[startup] {e}", flush=True); traceback.print_exc(); time.sleep(10); sys.exit(1)

    state = init_state(ex)
    last_summary = 0.0
    print(f"{COL_B}== LIVE GRID start =={COL_0} | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% | pairs={PAIRS} | buys={'ON' if ALLOW_BUYS else 'OFF'} | buffer=€{MIN_CASH_BUFFER_EUR:.2f} | keep_free=€{BUY_FREE_EUR_MIN:.2f}", flush=True)

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
                    qty = to_num(state["portfolio"]["coins"][p]["qty"],0.0)
                    live_inv += qty*mid
                    for l in state["pairs"][p]["inventory_lots"]:
                        invested_cost += l["qty"]*l["buy_price"]
                cap = cap_now(state["portfolio"])
                pnl = to_num(state["portfolio"]["pnl_realized"],0.0)
                print(f"[SUMMARY] total_eq=€{cash+live_inv:.2f} | cash=€{cash:.2f} | invested_cost=€{invested_cost:.2f} | live_inv=€{live_inv:.2f} | cap_now=€{cap:.2f} | pnl_realized=€{pnl:.2f} | buys={'ON' if ALLOW_BUYS else 'OFF'}", flush=True)
                write_equity(ex, state); save_state(state); last_summary = time.time()

            for line in out: print(line, flush=True)
            time.sleep(SLEEP_SEC)

        except ccxt.BaseError as e:
            print(f"[exchange] {e}", flush=True); time.sleep(2)
        except KeyboardInterrupt:
            print("Stop door gebruiker.", flush=True); save_state(state); break
        except Exception as e:
            print(f"[fatal] {e}", flush=True); traceback.print_exc(); time.sleep(2)

if __name__ == "__main__":
    main()
