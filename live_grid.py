# live_grid.py — Bitvavo grid-bot (safe & paste-and-run)
# ------------------------------------------------------
# - Werkt alleen met Bitvavo (via ccxt)
# - Hanteert operatorId (vereist door Bitvavo) — default "0000000000"
# - Koopt alleen als: ALLOW_BUYS=true én free_EUR >= BUY_FREE_EUR_MIN + buffer
# - Respecteert minimum orderregels (min quote/base)
# - Verkoopt zodra netto winst (incl. fees) >= MIN_PROFIT_EUR en pct >= MIN_PROFIT_PCT
# - Caps: CAPTIAL_EUR (plus realized PnL als REINVEST_PROFITS=true)
# - Schrijft logs + CSV’s in DATA_DIR

import os, sys, time, json, math, csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import ccxt

# ========= ENV / CONFIG =========
API_KEY     = os.getenv("API_KEY", "")
API_SECRET  = os.getenv("API_SECRET", "")
# Bitvavo vereist operatorId. Gebruik een vaste default als je er geen hebt.
OPERATOR_ID = os.getenv("OPERATOR_ID", "0000000000").strip() or "0000000000"

EXCHANGE    = os.getenv("EXCHANGE", "bitvavo").lower()
PAIRS       = [p.strip() for p in os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").split(",") if p.strip()]

CAPITAL_EUR        = float(os.getenv("CAPITAL_EUR", "1100"))
REINVEST_PROFITS   = os.getenv("REINVEST_PROFITS", "false").lower() in ("1","true","yes")

FEE_PCT            = float(os.getenv("FEE_PCT", "0.0015"))     # 0.15%
GRID_LEVELS        = int(os.getenv("GRID_LEVELS", "48"))
BAND_PCT           = float(os.getenv("BAND_PCT", "0.12"))      # 12% onder huidige prijs

MIN_PROFIT_PCT     = float(os.getenv("MIN_PROFIT_PCT", "0.0005"))
MIN_PROFIT_EUR     = float(os.getenv("MIN_PROFIT_EUR", "0.05")) # min € 0,05 netto
SELL_SAFETY_PCT    = float(os.getenv("SELL_SAFETY_PCT", "0.003"))

ORDER_SIZE_FACTOR  = float(os.getenv("ORDER_SIZE_FACTOR", "1.6"))

MIN_QUOTE_EUR      = float(os.getenv("MIN_QUOTE_EUR", "5"))
MIN_CASH_BUFFER_EUR= float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
BUY_FREE_EUR_MIN   = float(os.getenv("BUY_FREE_EUR_MIN", "100"))  # vrij € dat NIET aangeraakt mag worden
ALLOW_BUYS         = os.getenv("ALLOW_BUYS","true").lower() in ("1","true","yes")

LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE","false").lower() in ("1","true","yes")

# Wegingen per pair, bijv: "BTC/EUR:0.45,ETH/EUR:0.20,SOL/EUR:0.25,XRP/EUR:0.05,LTC/EUR:0.05"
WEIGHTS: Dict[str,float] = {}
for part in os.getenv("WEIGHTS","").split(","):
    if ":" in part:
        k,v = part.split(":",1)
        try: WEIGHTS[k.strip()] = float(v)
        except: pass
if WEIGHTS:
    s = sum(WEIGHTS.values()) or 1.0
    WEIGHTS = {k: v/s for k,v in WEIGHTS.items()}  # normaliseer

DATA_DIR   = Path(os.getenv("DATA_DIR","data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_JSON = DATA_DIR/"live_state.json"
TRADES_CSV = DATA_DIR/"live_trades.csv"
EQUITY_CSV = DATA_DIR/"live_equity.csv"

SLEEP_SEC        = int(os.getenv("SLEEP_SEC", "3"))
LOG_SUMMARY_SEC  = int(os.getenv("LOG_SUMMARY_SEC", "240"))

# ========= UI helpers =========
COL_G="\033[92m"; COL_R="\033[91m"; COL_C="\033[96m"; COL_B="\033[94m"; COL_RESET="\033[0m"
def now_iso(): return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
def to_num(x, d=0.0):
    try: return float(x)
    except: return d
def append_csv(path: Path, row: List[str], header: List[str] = None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header: w.writerow(header)
        w.writerow(row)
def load_json(p: Path, default):
    try:
        if p.exists(): return json.loads(p.read_text(encoding="utf-8"))
    except: pass
    return default
def save_json(p: Path, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

# ========= Exchange =========
def make_exchange():
    if EXCHANGE != "bitvavo":
        raise RuntimeError("Deze bot ondersteunt alleen Bitvavo (ccxt).")
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API_KEY en API_SECRET zijn verplicht.")
    ex = ccxt.bitvavo({
        "enableRateLimit": True,
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "options": {"operatorId": OPERATOR_ID},   # <<< Belangrijk
    })
    ex.load_markets()
    return ex

def fetch_mid(ex, pair):
    t = ex.fetch_ticker(pair)
    bid = to_num(t.get("bid"), 0.0)
    ask = to_num(t.get("ask"), 0.0)
    return (bid+ask)/2.0 if (bid and ask) else to_num(t.get("last"), 0.0)

def best_bid(ex, pair): return to_num(ex.fetch_ticker(pair).get("bid"), 0.0)
def best_ask(ex, pair): return to_num(ex.fetch_ticker(pair).get("ask"), 0.0)
def amount_to_precision(ex, pair, amt): return float(ex.amount_to_precision(pair, amt))

def min_rules(ex, pair, px_now) -> Tuple[float,float]:
    m = ex.markets.get(pair, {})
    min_base  = to_num(m.get("limits",{}).get("amount",{}).get("min"), 0.0) or 0.0
    min_quote = max(MIN_QUOTE_EUR, to_num(m.get("limits",{}).get("cost",{}).get("min"), 0.0) or 0.0)
    return (min_quote, min_base)

def free_eur(ex):  return to_num(ex.fetch_balance().get("EUR",{}).get("free"), 0.0)
def free_base(ex, base): return to_num(ex.fetch_balance().get(base,{}).get("free"), 0.0)

# ========= Grid helpers =========
def build_levels(px_now: float) -> List[float]:
    if px_now <= 0: return []
    span = px_now * BAND_PCT
    step = span / GRID_LEVELS
    return [px_now - step*i for i in range(1, GRID_LEVELS+1)]

def cap_now(state) -> float:
    pnl = to_num(state["portfolio"].get("pnl_realized", 0.0))
    return CAPITAL_EUR + (pnl if REINVEST_PROFITS else 0.0)

def weight(pair: str) -> float:
    if WEIGHTS: return WEIGHTS.get(pair, 0.0)
    return 1.0 / max(1, len(PAIRS))

def ticket_eur_for_pair(state, pair: str) -> float:
    alloc = cap_now(state) * weight(pair)
    base  = alloc / GRID_LEVELS
    return base * ORDER_SIZE_FACTOR

def target_sell(buy_price: float, qty: float) -> float:
    pct_target = buy_price * (1 + 2*FEE_PCT + MIN_PROFIT_PCT + SELL_SAFETY_PCT)
    abs_target = buy_price + (MIN_PROFIT_EUR / max(qty,1e-12)) * (1 + FEE_PCT)
    return max(pct_target, abs_target)

def net_gain_ok(buy_price: float, sell_price: float, qty: float) -> bool:
    gross = sell_price*qty
    net   = gross*(1 - FEE_PCT) - (buy_price*qty*(1 + FEE_PCT))
    pct   = (sell_price / buy_price) - 1.0
    return net >= MIN_PROFIT_EUR and pct >= MIN_PROFIT_PCT

# ========= IO / State =========
def init_state(ex):
    state = load_json(STATE_JSON, {
        "portfolio": {"pnl_realized": 0.0, "coins": {p: {"qty": 0.0} for p in PAIRS}},
        "baseline": {},
        "pairs": {p: {"last_price": 0.0, "levels": [], "inventory_lots": []} for p in PAIRS},
    })
    if LOCK_PREEXISTING_BALANCE and not state["baseline"]:
        bal = ex.fetch_balance()
        for p in PAIRS:
            base = p.split("/")[0]
            state["baseline"][base] = to_num(bal.get(base,{}).get("free"), 0.0)
    return state

def write_equity_row(ex, state):
    cash = free_eur(ex)
    val  = 0.0
    for p in PAIRS:
        q = to_num(state["portfolio"]["coins"].get(p,{}).get("qty"), 0.0)
        if q > 0: val += q * fetch_mid(ex, p)
    append_csv(EQUITY_CSV, [now_iso(), f"{cash+val:.2f}"], header=["timestamp","equity_eur"])

# ========= Order helpers =========
def buy_market(ex, pair, cost_eur):
    ask = best_ask(ex, pair)
    if ask <= 0: return (0.0, 0.0, 0.0, 0.0)
    qty = amount_to_precision(ex, pair, cost_eur/ask)
    if qty <= 0: return (0.0, 0.0, 0.0, 0.0)
    o = ex.create_order(pair, "market", "buy", qty)
    avg = to_num(o.get("average"), ask)
    filled = to_num(o.get("filled"), qty)
    cost = avg * filled
    fee  = cost * FEE_PCT
    return (filled, avg, fee, cost)

def sell_market(ex, pair, qty):
    bid = best_bid(ex, pair)
    if bid <= 0 or qty <= 0: return (0.0, 0.0, 0.0)
    o = ex.create_order(pair, "market", "sell", qty)
    avg = to_num(o.get("average"), bid)
    filled = to_num(o.get("filled"), qty)
    proceeds = avg * filled
    fee = proceeds * FEE_PCT
    return (proceeds, avg, fee)

# ========= Core per pair =========
def process_pair(ex, pair, state, out: List[str]):
    ps   = state["pairs"][pair]
    port = state["portfolio"]

    px_now = fetch_mid(ex, pair)
    if px_now <= 0:
        out.append(f"[{pair}] geen prijs; skip.")
        return

    if not ps["levels"]:
        ps["levels"] = build_levels(px_now)
    if ps["last_price"] <= 0:
        ps["last_price"] = px_now

    # --- SELL (FIFO)
    lots = ps["inventory_lots"]
    if lots:
        min_quote, min_base = min_rules(ex, pair, px_now)
        base_ccy = pair.split("/")[0]
        allowed_qty = None
        if LOCK_PREEXISTING_BALANCE:
            allowed_qty = max(0.0, free_base(ex, base_ccy) - to_num(state["baseline"].get(base_ccy,0.0)))

        changed = True
        while changed and ps["inventory_lots"]:
            changed = False
            lot = ps["inventory_lots"][0]
            trigger = target_sell(lot["buy_price"], lot["qty"])
            bid = best_bid(ex, pair)
            if bid < trigger:
                out.append(f"[{pair}] SELL wait: bid €{bid:.2f} < trigger €{trigger:.2f} (buy €{lot['buy_price']:.2f})")
                break

            qty = lot["qty"]
            if qty < min_base or (qty*px_now) < min_quote:
                out.append(f"[{pair}] SELL skip: lot te klein (amt {qty:.8f}/{min_base}, €{qty*px_now:.2f}/€{min_quote:.2f})")
                break
            if allowed_qty is not None and allowed_qty + 1e-12 < qty:
                out.append(f"[{pair}] SELL stop: baseline protect ({allowed_qty:.8f} {base_ccy} vrij).")
                break

            proceeds, avg, fee = sell_market(ex, pair, amount_to_precision(ex, pair, qty))
            if proceeds <= 0 or avg <= 0:
                out.append(f"[{pair}] SELL fail (geen fill)."); break

            pnl = proceeds - fee - (qty * lot["buy_price"] * (1 + FEE_PCT))
            port["pnl_realized"] = to_num(port.get("pnl_realized",0.0)) + pnl
            port["coins"].setdefault(pair,{"qty":0.0})
            port["coins"][pair]["qty"] = to_num(port["coins"][pair]["qty"]) - qty
            ps["inventory_lots"].pop(0)
            if allowed_qty is not None: allowed_qty -= qty

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "SELL", f"{avg:.6f}", f"{qty:.8f}", f"{proceeds:.2f}", "", pair.split("/")[0],
                 f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "take_profit"],
                header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"]
            )
            col = COL_G if pnl >= 0 else COL_R
            out.append(f"{col}[{pair}] SELL {qty:.8f} @ €{avg:.6f} | pnl=€{pnl:.2f} | trigger €{trigger:.2f}{COL_RESET}")
            changed = True

    # --- BUY (crossed levels)
    if not ALLOW_BUYS:
        out.append(f"[{pair}] BUY paused (ALLOW_BUYS=false).")
    else:
        crossed = [L for L in ps["levels"] if px_now < L <= ps["last_price"]]
        if crossed:
            freeE  = free_eur(ex)
            cap    = cap_now(state)
            invested = sum((l["qty"]*l["buy_price"]) for p in state["pairs"].values() for l in p["inventory_lots"])
            room   = cap - invested
            for L in crossed:
                min_quote, min_base = min_rules(ex, pair, px_now)
                ticket = ticket_eur_for_pair(state, pair)
                max_buyable = max(0.0, freeE - MIN_CASH_BUFFER_EUR)
                cost = min(ticket, max_buyable, room)
                cost = max(cost, min_quote)

                if freeE < BUY_FREE_EUR_MIN + MIN_CASH_BUFFER_EUR:
                    out.append(f"[{pair}] BUY skip: policy free_EUR<€{BUY_FREE_EUR_MIN:.2f} (buffer €{MIN_CASH_BUFFER_EUR:.2f}).")
                    continue
                if room <= 0:
                    out.append(f"[{pair}] BUY skip: cost-cap bereikt (cap_now=€{cap:.2f}).")
                    continue
                if cost <= 0:
                    need = max(min_quote + MIN_CASH_BUFFER_EUR - freeE, 0.0)
                    out.append(f"[{pair}] BUY skip: vrije EUR te laag (free≈€{freeE:.2f}, nodig≥€{min_quote+MIN_CASH_BUFFER_EUR:.2f}, tekort≈€{need:.2f}).")
                    continue

                qty, avg, fee, executed = buy_market(ex, pair, cost)
                if qty <= 0 or avg <= 0:
                    out.append(f"[{pair}] BUY fail: geen fill."); continue
                if qty < min_base or executed < min_quote:
                    out.append(f"[{pair}] BUY fill < minima (amt={qty:.8f}/{min_base}, €{executed:.2f}/€{min_quote:.2f})."); continue

                ps["inventory_lots"].append({"qty": qty, "buy_price": avg})
                port["coins"].setdefault(pair,{"qty":0.0})
                port["coins"][pair]["qty"] = to_num(port["coins"][pair]["qty"]) + qty

                tgt = target_sell(avg, qty)
                append_csv(
                    TRADES_CSV,
                    [now_iso(), pair, "BUY", f"{avg:.6f}", f"{qty:.8f}", f"{executed:.2f}", "", pair.split("/")[0],
                     f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
                    header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"]
                )
                out.append(f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avg:.6f} | exec≈€{executed:.2f} | → target SELL≈€{tgt:.2f}{COL_RESET}")

    ps["last_price"] = px_now

# ========= Main loop =========
def main():
    ex = make_exchange()
    state = init_state(ex)

    last_summary = 0.0
    print(f"== LIVE GRID start == | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% | "
          f"pairs={PAIRS} | buys={'ON' if ALLOW_BUYS else 'OFF'} | buffer=€{MIN_CASH_BUFFER_EUR:.2f} | keep_free=€{BUY_FREE_EUR_MIN:.2f}")

    while True:
        lines: List[str] = []
        try:
            for p in PAIRS:
                process_pair(ex, p, state, lines)

            # Periodieke samenvatting
            if (time.time() - last_summary) >= LOG_SUMMARY_SEC:
                cash = free_eur(ex)
                live_inv = 0.0
                invested = 0.0
                for p in PAIRS:
                    mid = fetch_mid(ex, p)
                    qty = to_num(state["portfolio"]["coins"][p]["qty"], 0.0)
                    live_inv += qty*mid
                    invested += sum(l["qty"]*l["buy_price"] for l in state["pairs"][p]["inventory_lots"])
                cap = cap_now(state)
                pnl = to_num(state["portfolio"]["pnl_realized"], 0.0)
                print(f"[SUMMARY] total_eq=€{cash+live_inv:.2f} | cash=€{cash:.2f} | free_EUR=€{cash:.2f} | "
                      f"invested_cost=€{invested:.2f} | live_inv=€{live_inv:.2f} | cap_now=€{cap:.2f} | "
                      f"pnl_realized=€{pnl:.2f} | buys={'ON' if ALLOW_BUYS else 'OFF'}")
                write_equity_row(ex, state)
                save_json(STATE_JSON, state)
                last_summary = time.time()

            for ln in lines: print(ln)
            time.sleep(SLEEP_SEC)

        except ccxt.BaseError as e:
            # Vaak tijdelijk (operatorId/ratelimit/netwerk). Log en door.
            print(f"[exchange] {str(e)}")
            time.sleep(2)
        except KeyboardInterrupt:
            print("Stop door gebruiker.")
            save_json(STATE_JSON, state)
            break
        except Exception as e:
            print(f"[fatal] {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
