# live_grid.py — compacte, veilige grid bot voor Bitvavo (ccxt)
# - Respecteert CAPITAL_EUR & cash buffer
# - Pauzeknop: ALLOW_BUYS=false
# - Extra koopdrempel: BUY_FREE_EUR_MIN
# - Locked baseline: LOCK_PREEXISTING_BALANCE (geen verkoop van pre-existente coins)
# - Minima/fees meegenomen; schrijft CSV's

import os, sys, time, math, json, csv, traceback
from pathlib import Path
from datetime import datetime, timezone
import ccxt

# ================== ENV ==================
API_KEY      = os.getenv("API_KEY","").strip()
API_SECRET   = os.getenv("API_SECRET","").strip()
EXCHANGE     = os.getenv("EXCHANGE","bitvavo").lower()
COINS        = [s.strip().upper() for s in os.getenv("COINS","BTC/EUR,ETH/EUR").split(",") if s.strip()]
WEIGHTS_STR  = os.getenv("WEIGHTS","")
CAPITAL_EUR  = float(os.getenv("CAPITAL_EUR","1100"))
FEE_PCT      = float(os.getenv("FEE_PCT","0.0015"))      # 0.15% Bitvavo (maker/taker dicht bij 0.15–0.25%)
GRID_LEVELS  = int(os.getenv("GRID_LEVELS","32"))
BAND_PCT     = float(os.getenv("BAND_PCT","0.12"))       # grid band als fractie van prijs
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR","1.8"))  # snellere afbouw laag
MIN_PROFIT_PCT    = float(os.getenv("MIN_PROFIT_PCT","0.0005"))  # 0.05%
MIN_PROFIT_EUR    = float(os.getenv("MIN_PROFIT_EUR","0.05"))
SELL_SAFETY_PCT   = float(os.getenv("SELL_SAFETY_PCT","0.003"))  # 0.3% extra t.o.v. buy→target controle
MIN_QUOTE_EUR     = float(os.getenv("MIN_QUOTE_EUR","5"))        # Bitvavo min. ~€5
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR","25"))
REINVEST_PROFITS  = os.getenv("REINVEST_PROFITS","false").lower() in ("1","true","yes")
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE","").strip()
OPERATOR_ID       = os.getenv("OPERATOR_ID","00000000")

# Pauzeknop & extra koopdrempel
ALLOW_BUYS        = os.getenv("ALLOW_BUYS","true").lower() in ("1","true","yes")
BUY_FREE_EUR_MIN  = float(os.getenv("BUY_FREE_EUR_MIN","0"))

DATA_DIR  = Path(os.getenv("DATA_DIR","data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
TRADES_CSV = DATA_DIR / "live_trades.csv"
EQUITY_CSV = DATA_DIR / "live_equity.csv"
STATE_JSON = DATA_DIR / "live_state.json"

# ================== Helpers ==================
COL_R="\033[91m"; COL_G="\033[92m"; COL_C="\033[96m"; COL_B="\033[94m"; COL_RESET="\033[0m"

def now_iso():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def append_csv(path: Path, row, header=None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header: w.writerow(header)
        w.writerow(row)

def save_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def load_json(path: Path, default):
    if not path.exists(): return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def to_weights(symbols, weights_str):
    # "BTC/EUR:0.45,ETH/EUR:0.20" → dict
    d = {}
    for token in weights_str.split(","):
        token = token.strip()
        if not token: continue
        if ":" in token:
            k, v = token.split(":",1)
            try: d[k.strip().upper()] = float(v)
            except: pass
    if not d:
        # gelijk verdelen
        w = 1.0/len(symbols)
        return {s:w for s in symbols}
    # normaliseren
    s = sum(max(0.0, x) for x in d.values()) or 1.0
    return {k:max(0.0,v)/s for k,v in d.items() if k in symbols}

def euro_per_ticket(alloc_eur, n_levels):
    # Ticketgrootte per stap (iets agressiever met ORDER_SIZE_FACTOR)
    base = alloc_eur / max(1, n_levels)
    return base * ORDER_SIZE_FACTOR

def target_sell_price(buy_px):
    # Minimale netto winst + veiligheidsmarge
    return buy_px * (1 + MIN_PROFIT_PCT + SELL_SAFETY_PCT) + MIN_PROFIT_EUR

def net_gain_ok(buy_px, sell_px, fee, min_pct, min_eur, qty):
    proceeds = sell_px * qty
    cost     = buy_px * qty
    # fees aan beide kanten
    proceeds_net = proceeds * (1 - fee)
    cost_gross   = cost * (1 + fee)
    pnl = proceeds_net - cost_gross
    cond_pct = (sell_px >= buy_px * (1 + min_pct))
    return (pnl >= min_eur) and cond_pct

def amount_to_precision(ex, symbol, qty):
    try:
        info = ex.market(symbol)
        p = info.get("precision",{}).get("amount", None)
        if p is None:
            step = info.get("limits",{}).get("amount",{}).get("min", 0.0) or 0.0
            if step and step>0: return math.floor(qty/step)*step
            return qty
        fmt = "{:0." + str(p) + "f}"
        return float(fmt.format(qty))
    except Exception:
        return qty

def price_to_precision(ex, symbol, price):
    try:
        info = ex.market(symbol)
        p = info.get("precision",{}).get("price", None)
        if p is None: return price
        fmt = "{:0." + str(p) + "f}"
        return float(fmt.format(price))
    except Exception:
        return price

# ================== Exchange ==================
def build_exchange():
    if EXCHANGE != "bitvavo":
        raise RuntimeError("Deze bot is ingesteld voor Bitvavo (EXCHANGE=bitvavo).")
    if not API_KEY or not API_SECRET:
        raise RuntimeError("API_KEY/API_SECRET ontbreken.")
    ex = ccxt.bitvavo({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    ex.load_markets()
    return ex

def best_bid_px(ex, symbol):
    ob = ex.fetch_order_book(symbol, 5)
    if ob["bids"]: return float(ob["bids"][0][0])
    return float(ex.fetch_ticker(symbol)["last"])

def best_ask_px(ex, symbol):
    ob = ex.fetch_order_book(symbol, 5)
    if ob["asks"]: return float(ob["asks"][0][0])
    return float(ex.fetch_ticker(symbol)["last"])

def market_mins(ex, symbol, price_now):
    info = ex.market(symbol)
    min_base = info.get("limits",{}).get("amount",{}).get("min", 0.0) or 0.0
    min_quote= info.get("limits",{}).get("cost",{}).get("min", MIN_QUOTE_EUR) or MIN_QUOTE_EUR
    # zorg dat min_quote ≥ €5
    min_quote = max(min_quote, MIN_QUOTE_EUR)
    return (float(min_quote), float(min_base))

def free_eur_on_exchange(ex):
    bal = ex.fetch_balance()
    return float(bal.get("EUR",{}).get("free", 0.0) or 0.0)

def free_base_on_exchange(ex, base_symbol):
    bal = ex.fetch_balance()
    return float(bal.get(base_symbol,{}).get("free", 0.0) or 0.0)

def buy_market(ex, symbol, cost_eur):
    # koop met kostenmarge
    ask = best_ask_px(ex, symbol)
    min_quote, min_base = market_mins(ex, symbol, ask)
    cost_eur = max(cost_eur, min_quote)
    qty = cost_eur / max(1e-12, ask)
    qty = amount_to_precision(ex, symbol, qty)
    if qty <= 0: return 0.0, 0.0, 0.0, 0.0
    order = ex.create_order(symbol, "market", "buy", qty)
    executed = sum((float(f["cost"]) for f in order.get("fills", []) if "cost" in f), 0.0)
    fee      = sum((float(f["fee"]["cost"]) for f in order.get("fills", []) if f.get("fee")), 0.0)
    avg_px   = float(order.get("average") or (executed/max(qty,1e-12)))
    return float(order.get("filled", qty)), avg_px, fee, executed

def sell_market(ex, symbol, qty):
    bid = best_bid_px(ex, symbol)
    qty = amount_to_precision(ex, symbol, qty)
    if qty <= 0: return 0.0, 0.0, 0.0
    order = ex.create_order(symbol, "market", "sell", qty)
    proceeds= sum((float(f["cost"]) for f in order.get("fills", []) if "cost" in f), 0.0)
    fee     = sum((float(f["fee"]["cost"]) for f in order.get("fills", []) if f.get("fee")), 0.0)
    avg_px  = float(order.get("average") or (proceeds/max(qty,1e-12)))
    return proceeds, avg_px, fee

# ================== GRID STATE ==================
def init_state(ex):
    # Baseline vastleggen (lock preexisting)
    baseline = {}
    if LOCK_PREEXISTING_BALANCE:
        bal = ex.fetch_balance()
        for sym in COINS:
            base = sym.split("/")[0]
            baseline[base] = float(bal.get(base,{}).get("free",0.0) or 0.0)

    weights = to_weights(COINS, WEIGHTS_STR)
    per_coin_cap = {s: CAPITAL_EUR * float(weights.get(s, 0.0)) for s in COINS}
    state = {
        "operator": OPERATOR_ID,
        "created": now_iso(),
        "baseline": baseline,  # locked
        "pnl_realized": 0.0,
        "portfolio": {
            "cash_eur": free_eur_on_exchange(ex),
            "coins": {s: {"qty":0.0, "cash_alloc": per_coin_cap[s]} for s in COINS}
        },
        "per_coin_cap": per_coin_cap,
        "inventory": {s: [] for s in COINS},  # lijst lots: {"qty":.., "buy":..}
        "grid_levels": GRID_LEVELS,
        "band_pct": BAND_PCT,
        "last_price": {s: None for s in COINS},
    }
    return state

def cap_now(state):
    return CAPITAL_EUR

def invested_cost_eur(state):
    total = 0.0
    for s in COINS:
        for l in state["inventory"][s]:
            total += l["qty"] * l["buy"]
    return total

def bot_inventory_value_eur_from_exchange(ex, state):
    total = 0.0
    bal = ex.fetch_balance()
    for s in COINS:
        base = s.split("/")[0]
        qty = float(bal.get(base,{}).get("free",0.0) or 0.0)
        px  = best_bid_px(ex, s)
        # exclusief baseline
        base_locked = float(state.get("baseline",{}).get(base,0.0) or 0.0)
        qty_bot = max(0.0, qty - base_locked)
        total += qty_bot * px
    return total

def make_levels(price, n, band_pct):
    lo = price * (1 - band_pct/2)
    hi = price * (1 + band_pct/2)
    step = (hi - lo)/max(1,n)
    return [lo + i*step for i in range(1, n+1)]

# ================== LOGICA ==================
def try_pair(ex, pair, state, logs):
    port = state["portfolio"]
    inv  = state["inventory"][pair]
    last = state["last_price"][pair]
    ask  = best_ask_px(ex, pair)
    bid  = best_bid_px(ex, pair)

    # Levels per pair dynamisch rond current price
    levels = make_levels(ask, state["grid_levels"], state["band_pct"])

    # -------- BUY --------
    if not ALLOW_BUYS:
        logs.append(f"[{pair}] BUY paused (ALLOW_BUYS=false).")
    elif last is not None and ask < last:
        crossed = [L for L in levels if ask < L <= last]
        avail = free_eur_on_exchange(ex)
        if avail < (BUY_FREE_EUR_MIN + MIN_CASH_BUFFER_EUR):
            tekort = (BUY_FREE_EUR_MIN + MIN_CASH_BUFFER_EUR) - avail
            if crossed:
                L = crossed[0]
                logs.append(f"[{pair}] BUY skip: vrije EUR te laag (free≈€{avail:.2f}, nodig≥€{BUY_FREE_EUR_MIN+MIN_CASH_BUFFER_EUR:.2f}, tekort≈€{tekort:.2f}). PLAN @≈€{L:.2f} → SELL≈€{target_sell_price(L):.2f}")
        else:
            for L in crossed:
                # alloc/ticket
                ticket = euro_per_ticket(port["coins"][pair]["cash_alloc"], state["grid_levels"])
                max_cost = max(0.0, avail - MIN_CASH_BUFFER_EUR) / (1.0 + FEE_PCT)
                cost    = min(ticket, max_cost)

                min_quote, min_base = market_mins(ex, pair, ask)
                need = max(min_quote, min_base*ask)
                cost = max(cost, need)
                cost = math.ceil(cost * 1.01)

                # cap checks
                live_cap = bot_inventory_value_eur_from_exchange(ex, state)
                cap = cap_now(state)
                if invested_cost_eur(state) + cost > cap + 1e-6:
                    logs.append(f"[{pair}] BUY skip: cost-cap (cap=€{cap:.2f}). PLAN @≈€{L:.2f} → SELL≈€{target_sell_price(L):.2f}")
                    continue
                if live_cap + cost > cap + 1e-6:
                    logs.append(f"[{pair}] BUY skip: live-cap (≈€{live_cap:.2f}/{cap:.2f}). PLAN @≈€{L:.2f} → SELL≈€{target_sell_price(L):.2f}")
                    continue

                # koop
                qty, avg, fee, executed = buy_market(ex, pair, cost)
                if qty <= 0 or avg <= 0:
                    logs.append(f"[{pair}] BUY fail: geen fill. PLAN @≈€{L:.2f} → SELL≈€{target_sell_price(L):.2f}")
                    continue
                # minima check
                if qty < min_base or executed < min_quote:
                    logs.append(f"[{pair}] BUY fill < minima (amt={qty:.8f} / {min_base}, €{executed:.2f} / €{min_quote:.2f}); skip.")
                    continue

                inv.append({"qty": qty, "buy": avg})
                port["cash_eur"] -= (executed + fee)
                append_csv(TRADES_CSV,
                           [now_iso(), pair, "BUY", f"{avg:.6f}", f"{qty:.8f}", f"{executed:.2f}", f"{port['cash_eur']:.2f}",
                            pair.split("/")[0], f"{sum(l['qty'] for l in inv):.8f}", "", "grid_buy"],
                           header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"])
                logs.append(f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avg:.6f} → target SELL≈€{target_sell_price(avg):.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}")

    # -------- SELL --------
    if inv:
        base = pair.split("/")[0]
        bot_free = None
        if state.get("baseline"):
            bot_free = max(0.0, free_base_on_exchange(ex, base) - float(state["baseline"].get(base, 0.0)))

        changed = True
        while changed and inv:
            changed = False
            # zoek lot dat target haalt
            idx = next((i for i,l in enumerate(inv)
                        if net_gain_ok(l["buy"], bid, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, l["qty"])),
                       None)
            if idx is None:
                # toon top wait
                top = inv[0]
                trig = target_sell_price(top["buy"])
                logs.append(f"[{pair}] SELL wait: bid €{bid:.2f} < trigger €{trig:.2f} (buy €{top['buy']:.2f})")
                break

            lot = inv[idx]
            min_quote, min_base = market_mins(ex, pair, bid)
            if lot["qty"] < min_base or lot["qty"]*bid < min_quote:
                logs.append(f"[{pair}] SELL skip: lot te klein (amt {lot['qty']:.8f}/{min_base} of €{lot['qty']*bid:.2f}/€{min_quote:.2f})")
                break

            if bot_free is not None and bot_free + 1e-12 < lot["qty"]:
                logs.append(f"[{pair}] SELL stop (baseline protect): vrij {bot_free:.8f} {base}")
                break

            proceeds, avg, fee = sell_market(ex, pair, lot["qty"])
            if proceeds > 0 and avg > 0:
                pnl = (proceeds - fee) - (lot["qty"]*lot["buy"]*(1+FEE_PCT))
                inv.pop(idx)
                port["cash_eur"] += (proceeds - fee)
                state["pnl_realized"] += pnl
                if bot_free is not None: bot_free -= lot["qty"]
                append_csv(TRADES_CSV,
                           [now_iso(), pair, "SELL", f"{avg:.6f}", f"{lot['qty']:.8f}", f"{proceeds:.2f}", f"{port['cash_eur']:.2f}",
                            base, f"{sum(l['qty'] for l in inv):.8f}", f"{pnl:.2f}", "take_profit"])
                col = COL_G if pnl >= 0 else COL_R
                logs.append(f"{col}[{pair}] SELL {lot['qty']:.8f} @ €{avg:.6f} | pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}")
                changed = True

    state["last_price"][pair] = ask

def equity_snapshot(ex, state):
    cash = free_eur_on_exchange(ex)
    inv  = 0.0
    for s in COINS:
        base = s.split("/")[0]
        free_base = free_base_on_exchange(ex, base)
        px = best_bid_px(ex, s)
        base_locked = float(state.get("baseline",{}).get(base,0.0) or 0.0)
        inv += max(0.0, free_base - base_locked) * px
    total = cash + inv
    append_csv(EQUITY_CSV, [now_iso(), f"{total:.2f}"], header=["timestamp","equity_eur"])
    return total, cash, inv

# ================== MAIN LOOP ==================
def main():
    print(f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={FEE_PCT:.3%} | pairs={COINS} | "
          f"min_profit={MIN_PROFIT_PCT:.2%} / €{MIN_PROFIT_EUR:.2f} | sell_safety={SELL_SAFETY_PCT:.2%} | buffer=€{MIN_CASH_BUFFER_EUR:.2f}")
    ex = build_exchange()

    state = load_json(STATE_JSON, None) or init_state(ex)
    save_json(STATE_JSON, state)

    last_sum = 0.0
    while True:
        try:
            logs=[]
            for p in COINS:
                try_pair(ex, p, state, logs)

            total, cash, inv = equity_snapshot(ex, state)
            cap_now_eur = cap_now(state)
            msg = (f"[SUMMARY] total_eq=€{total:.2f} | cash=€{cash:.2f} | free_EUR≥€{BUY_FREE_EUR_MIN:.2f} "
                   f"| invested_cost=€{invested_cost_eur(state):.2f} | live_inv=€{inv:.2f} | cap_now=€{cap_now_eur:.2f} "
                   f"| pnl_realized=€{state['pnl_realized']:.2f} | buys={'ON' if ALLOW_BUYS else 'OFF'}")
            print(msg)

            for line in logs:
                print(line)

            save_json(STATE_JSON, state)
            # heartbeat elke ~60s
            time.sleep(60)
        except ccxt.NetworkError as e:
            print(f"{COL_R}[netwerk] {e}{COL_RESET}"); time.sleep(5)
        except ccxt.ExchangeError as e:
            print(f"{COL_R}[exchange] {e}{COL_RESET}"); time.sleep(5)
        except KeyboardInterrupt:
            print("Stop."); break
        except Exception as e:
            print(f"{COL_R}[error] {e}{COL_RESET}")
            traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    main()
