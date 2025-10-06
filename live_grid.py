# live_grid.py — Bitvavo grid-bot (compact & veilig)
import os, sys, time, json, math, csv
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List
import ccxt

# ---------- ENV ----------
API_KEY  = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
OPERATOR_ID = os.getenv("OPERATOR_ID", "").strip() or None

EXCHANGE = os.getenv("EXCHANGE", "bitvavo").lower()
PAIRS = [p.strip() for p in os.getenv("COINS","BTC/EUR,ETH/EUR").split(",") if p.strip()]

CAPITAL_EUR = float(os.getenv("CAPITAL_EUR","1100"))
REINVEST_PROFITS = os.getenv("REINVEST_PROFITS","false").lower() in ("1","true","yes")

FEE_PCT = float(os.getenv("FEE_PCT","0.0015"))
GRID_LEVELS = int(os.getenv("GRID_LEVELS","32"))
BAND_PCT = float(os.getenv("BAND_PCT","0.12"))  # totale band (12%) onder actuele prijs

MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT","0.0005"))
MIN_PROFIT_EUR = float(os.getenv("MIN_PROFIT_EUR","0.05"))
SELL_SAFETY_PCT = float(os.getenv("SELL_SAFETY_PCT","0.003"))  # extra marge bij verkopen

ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR","1.6"))

MIN_QUOTE_EUR = float(os.getenv("MIN_QUOTE_EUR","5"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR","25"))
BUY_FREE_EUR_MIN = float(os.getenv("BUY_FREE_EUR_MIN","100"))
ALLOW_BUYS = os.getenv("ALLOW_BUYS","true").lower() in ("1","true","yes")

LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE","false").lower() in ("1","true","yes")

WEIGHTS = {}
for part in os.getenv("WEIGHTS","").split(","):
    if ":" in part:
        k,v = part.split(":",1)
        try: WEIGHTS[k.strip()] = float(v)
        except: pass
if WEIGHTS:
    s = sum(WEIGHTS.values()) or 1.0
    WEIGHTS = {k: v/s for k,v in WEIGHTS.items()}

DATA_DIR = Path(os.getenv("DATA_DIR","data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_JSON   = DATA_DIR/"live_state.json"
TRADES_CSV   = DATA_DIR/"live_trades.csv"
EQUITY_CSV   = DATA_DIR/"live_equity.csv"

SLEEP_SEC = int(os.getenv("SLEEP_SEC","3"))
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC","240"))

# ---------- Utils ----------
COL_G="\033[92m"; COL_R="\033[91m"; COL_C="\033[96m"; COL_RESET="\033[0m"

def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

def load_json(p: Path, default):
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except: pass
    return default

def save_json(p: Path, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)

def append_csv(path: Path, row: List[str], header: List[str] = None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header: w.writerow(header)
        w.writerow(row)

def to_num(x, d=0.0):
    try: return float(x)
    except: return d

# ---------- Exchange ----------
def make_exchange():
    if EXCHANGE != "bitvavo":
        raise RuntimeError("Deze bot is gebouwd voor Bitvavo via ccxt.")
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API_KEY en API_SECRET vereist.")
    opts = {"enableRateLimit": True, "apiKey": API_KEY, "secret": API_SECRET}
    if OPERATOR_ID:
        opts["options"] = {"operatorId": OPERATOR_ID}
    ex = ccxt.bitvavo(opts)
    ex.load_markets()
    return ex

def fetch_mid_price(ex, pair):
    t = ex.fetch_ticker(pair)
    bid = to_num(t.get("bid"), 0.0)
    ask = to_num(t.get("ask"), 0.0)
    if bid and ask: return (bid+ask)/2.0
    return to_num(t.get("last"), 0.0)

def best_bid_px(ex, pair):
    t = ex.fetch_ticker(pair)
    return to_num(t.get("bid"), 0.0)

def best_ask_px(ex, pair):
    t = ex.fetch_ticker(pair)
    return to_num(t.get("ask"), 0.0)

def amount_to_precision(ex, pair, amt):
    return float(ex.amount_to_precision(pair, amt))

def price_to_precision(ex, pair, px):
    return float(ex.price_to_precision(pair, px))

# ---------- Portfolio helpers ----------
def free_eur(ex):
    bal = ex.fetch_balance()
    return to_num(bal.get("EUR",{}).get("free"), 0.0)

def free_base(ex, base):
    bal = ex.fetch_balance()
    return to_num(bal.get(base,{}).get("free"), 0.0)

def min_mkt_rules(ex, pair, px_now):
    # ccxt market info
    m = ex.markets.get(pair, {})
    min_base = to_num(m.get("limits",{}).get("amount",{}).get("min"), 0.0) or 0.0
    min_quote = max(MIN_QUOTE_EUR, to_num(m.get("limits",{}).get("cost",{}).get("min"), 0.0) or 0.0)
    return (min_quote, min_base)

# ---------- Grid helpers ----------
def build_levels(px_now: float) -> List[float]:
    if px_now <= 0: return []
    # GRID_LEVELS niveaus, gelijkmatig verdeeld over BAND_PCT onder huidige prijs
    span = px_now * BAND_PCT
    step = span / GRID_LEVELS
    return [px_now - step * i for i in range(1, GRID_LEVELS+1)]

def target_sell_price(buy_price: float, qty: float) -> float:
    # prijs zodat: opbrengst*(1-fee) - cost*(1+fee) >= MIN_PROFIT_EUR & pct >= MIN_PROFIT_PCT
    # benadering: + (2*fee + MIN_PROFIT_PCT + SELL_SAFETY_PCT) en absolute winst
    pct_target = buy_price * (1 + (2*FEE_PCT) + MIN_PROFIT_PCT + SELL_SAFETY_PCT)
    abs_target = buy_price + (MIN_PROFIT_EUR / max(qty,1e-12)) * (1 + FEE_PCT)
    return max(pct_target, abs_target)

def net_gain_ok(buy_price, sell_price, qty):
    gross = sell_price*qty
    net = gross*(1 - FEE_PCT) - (buy_price*qty*(1 + FEE_PCT))
    pct = (sell_price / buy_price) - 1.0
    return net >= MIN_PROFIT_EUR and pct >= MIN_PROFIT_PCT

def pair_weight(pair):
    if WEIGHTS:
        return WEIGHTS.get(pair, 0.0)
    return 1.0/len(PAIRS)

def cap_now(state):
    pnl = to_num(state["portfolio"].get("pnl_realized",0.0), 0.0)
    return CAPITAL_EUR + (pnl if REINVEST_PROFITS else 0.0)

def ticket_eur_for_pair(state, pair):
    alloc = cap_now(state) * pair_weight(pair)
    # Portionering over levels & order_size_factor
    base = alloc / GRID_LEVELS
    return base * ORDER_SIZE_FACTOR

# ---------- IO state ----------
def init_state(ex):
    state = load_json(STATE_JSON, {
        "portfolio":{
            "pnl_realized": 0.0,
            "cash_eur": free_eur(ex),
            "coins": {p: {"qty": 0.0} for p in PAIRS}
        },
        "baseline": {},  # voor lock preexisting
        "pairs": {p: {"last_price": 0.0, "levels": [], "inventory_lots": []} for p in PAIRS}
    })
    if LOCK_PREEXISTING_BALANCE and not state["baseline"]:
        bal = ex.fetch_balance()
        base = {p.split("/")[0]: to_num(bal.get(p.split("/")[0],{}).get("free"), 0.0) for p in PAIRS}
        state["baseline"] = base
    return state

def write_equity_row(ex, state):
    # equity = cash + waarde posities (bot-qty) tegen mid
    cash = free_eur(ex)
    value = 0.0
    for p in PAIRS:
        base = p.split("/")[0]
        bot_qty = to_num(state["portfolio"]["coins"].get(p,{}).get("qty"),0.0)
        if bot_qty > 0:
            mid = fetch_mid_price(ex, p)
            value += bot_qty * mid
    eq = cash + value
    append_csv(EQUITY_CSV, [now_iso(), f"{eq:.2f}"], header=["timestamp","equity_eur"])

# ---------- Trading actions ----------
def buy_market(ex, pair, cost_eur):
    ask = best_ask_px(ex, pair)
    if ask <= 0: return (0.0, 0.0, 0.0, 0.0)
    qty = amount_to_precision(ex, pair, cost_eur / ask)
    if qty <= 0: return (0.0, 0.0, 0.0, 0.0)
    o = ex.create_order(pair, "market", "buy", qty)
    avg = to_num(o.get("average"), ask)
    filled = to_num(o.get("filled"), qty)
    cost = avg * filled
    fee = cost * FEE_PCT
    return (filled, avg, fee, cost)

def sell_market(ex, pair, qty):
    bid = best_bid_px(ex, pair)
    if bid <= 0 or qty <= 0: return (0.0, 0.0, 0.0)
    o = ex.create_order(pair, "market", "sell", qty)
    avg = to_num(o.get("average"), bid)
    proceeds = avg * to_num(o.get("filled"), qty)
    fee = proceeds * FEE_PCT
    return (proceeds, avg, fee)

# ---------- Core per pair ----------
def process_pair(ex, pair, state, logs: List[str]):
    port = state["portfolio"]
    ps = state["pairs"][pair]

    px_now = fetch_mid_price(ex, pair)
    if px_now <= 0:
        logs.append(f"[{pair}] geen prijs, skip.")
        return

    # init levels
    if not ps["levels"]:
        ps["levels"] = build_levels(px_now)
    if ps["last_price"] <= 0:
        ps["last_price"] = px_now

    # ---- SELL: check lot targets
    lots = ps["inventory_lots"]
    if lots:
        min_quote, min_base = min_mkt_rules(ex, pair, px_now)
        base_ccy = pair.split("/")[0]
        allowed_qty = None
        if LOCK_PREEXISTING_BALANCE:
            free_b = free_base(ex, base_ccy)
            base_line = to_num(state["baseline"].get(base_ccy,0.0), 0.0)
            allowed_qty = max(0.0, free_b - base_line)

        changed = True
        while changed and ps["inventory_lots"]:
            changed = False
            # pak eerst het oudste lot (FIFO)
            lot = ps["inventory_lots"][0]
            trigger = target_sell_price(lot["buy_price"], lot["qty"])
            bid = best_bid_px(ex, pair)
            if bid < trigger:
                logs.append(f"[{pair}] SELL wait: bid €{bid:.2f} < trigger €{trigger:.2f} (buy €{lot['buy_price']:.2f})")
                break

            qty = lot["qty"]
            if qty < min_base or (qty*px_now) < min_quote:
                logs.append(f"[{pair}] SELL skip: lot te klein (amt {qty:.8f} / min {min_base}, €{qty*px_now:.2f} / min €{min_quote:.2f})")
                break
            if allowed_qty is not None and allowed_qty + 1e-12 < qty:
                logs.append(f"[{pair}] SELL stop: baseline protect ({allowed_qty:.8f} {base_ccy} vrij).")
                break

            proceeds, avg, fee = sell_market(ex, pair, amount_to_precision(ex, pair, qty))
            if proceeds <= 0 or avg <= 0:
                logs.append(f"[{pair}] SELL fail (geen fill)."); break

            pnl = proceeds - fee - (qty * lot["buy_price"] * (1 + FEE_PCT))
            port["pnl_realized"] = to_num(port.get("pnl_realized",0.0)) + pnl
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
            logs.append(f"{col}[{pair}] SELL {qty:.8f} @ €{avg:.6f} | pnl=€{pnl:.2f} | trigger €{trigger:.2f}{COL_RESET}")
            changed = True

    # ---- BUY: check cross naar lagere level
    if not ALLOW_BUYS:
        logs.append(f"[{pair}] BUY paused (ALLOW_BUYS=false).")
    else:
        # crossed levels?
        crossed = [L for L in ps["levels"] if px_now < L <= ps["last_price"]]
        if crossed:
            freeE = free_eur(ex)
            cap = cap_now(state)
            invested = sum((l["qty"]*l["buy_price"]) for p in state["pairs"].values() for l in p["inventory_lots"])
            room = cap - invested
            for L in crossed:
                min_quote, min_base = min_mkt_rules(ex, pair, px_now)

                # ticket sizing
                ticket = ticket_eur_for_pair(state, pair)
                max_buyable = max(0.0, freeE - MIN_CASH_BUFFER_EUR)
                cost = min(ticket, max_buyable, room)
                # respecteer min-quote
                cost = max(cost, min_quote)
                if cost <= 0:
                    need = max(min_quote + MIN_CASH_BUFFER_EUR - freeE, 0.0)
                    logs.append(f"[{pair}] BUY skip: vrije EUR te laag (free≈€{freeE:.2f}, nodig≥€{min_quote+MIN_CASH_BUFFER_EUR:.2f}, tekort≈€{need:.2f}).")
                    continue
                if freeE < BUY_FREE_EUR_MIN + MIN_CASH_BUFFER_EUR:
                    logs.append(f"[{pair}] BUY skip: policy free_EUR<€{BUY_FREE_EUR_MIN:.2f} (buffer €{MIN_CASH_BUFFER_EUR:.2f}).")
                    continue
                if room <= 0:
                    logs.append(f"[{pair}] BUY skip: cost-cap bereikt (cap=€{cap:.2f}).")
                    continue

                qty, avg, fee, executed = buy_market(ex, pair, cost)
                if qty <= 0 or avg <= 0:
                    logs.append(f"[{pair}] BUY fail: geen fill."); continue
                if qty < min_base or executed < min_quote:
                    logs.append(f"[{pair}] BUY fill < minima (amt={qty:.8f} / {min_base}, €{executed:.2f} / €{min_quote:.2f})."); continue

                # boek lot
                ps["inventory_lots"].append({"qty": qty, "buy_price": avg})
                port["coins"][pair]["qty"] = to_num(port["coins"][pair]["qty"]) + qty

                tgt = target_sell_price(avg, qty)
                append_csv(
                    TRADES_CSV,
                    [now_iso(), pair, "BUY", f"{avg:.6f}", f"{qty:.8f}", f"{executed:.2f}", "", pair.split("/")[0],
                     f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
                    header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"]
                )
                logs.append(f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avg:.6f} | exec≈€{executed:.2f} | → target SELL≈€{tgt:.2f}{COL_RESET}")

    ps["last_price"] = px_now

# ---------- Main loop ----------
def main():
    ex = make_exchange()
    state = init_state(ex)

    last_summary = 0.0
    print("== LIVE GRID start ==",
          f"capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% | pairs={PAIRS} | buys={'ON' if ALLOW_BUYS else 'OFF'}")

    while True:
        logs: List[str] = []
        try:
            for p in PAIRS:
                process_pair(ex, p, state, logs)
            # samenvatting
            if (time.time() - last_summary) >= LOG_SUMMARY_SEC:
                cap = cap_now(state)
                cash = free_eur(ex)
                invested = sum((l["qty"]*l["buy_price"]) for ps in state["pairs"].values() for l in ps["inventory_lots"])
                live_inv = 0.0
                for p in PAIRS:
                    mid = fetch_mid_price(ex, p)
                    qty = to_num(state["portfolio"]["coins"][p]["qty"],0.0)
                    live_inv += qty*mid
                pnl = to_num(state["portfolio"]["pnl_realized"],0.0)
                print(f"[SUMMARY] total_eq=€{cash+live_inv:.2f} | cash=€{cash:.2f} | free_EUR=€{cash:.2f} | invested_cost=€{invested:.2f} | live_inv=€{live_inv:.2f} | cap_now=€{cap:.2f} | pnl_realized=€{pnl:.2f} | buys={'ON' if ALLOW_BUYS else 'OFF'}")
                write_equity_row(ex, state)
                save_json(STATE_JSON, state)
                last_summary = time.time()

            # print logs
            for line in logs: print(line)
            time.sleep(SLEEP_SEC)
        except ccxt.BaseError as e:
            # vaak tijdelijk (rate limit / network / operatorId). log en ga verder
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
