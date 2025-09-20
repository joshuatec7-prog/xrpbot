# live_grid.py — Bitvavo LIVE grid bot
# - Kóópt alleen binnen je ingestelde CAP en alléén als er genoeg free EUR is (geen 216-errors meer)
# - Verkoopt op take-profit mét veiligheidsmarge op BEST BID
# - Optionele stop-loss
# - Logt altijd: koopprijs, TP (trigger), wachtregels (bid vs trigger) en werkelijke verkoopprijs
# - Schrijft naar data/live_trades.csv en data/live_equity.csv

import csv, json, os, time, math, random
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

# ========= helpers =========
COL_G="\033[92m"; COL_R="\033[91m"; COL_C="\033[96m"; COL_RESET="\033[0m"
def now_iso(): return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
def pct(x): return f"{x*100:.2f}%"

def append_csv(path: Path, row, header=None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header: w.writerow(header)
        w.writerow(row)

def load_json(path: Path, default):
    try:
        if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    except Exception: pass
    return default

def save_json(path: Path, obj):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

# ========= ENV =========
def _b(s, default="false"): return os.getenv(s, default).lower() in ("1","true","yes","on")

API_KEY  = os.getenv("API_KEY","")
API_SECRET = os.getenv("API_SECRET","")
EXCHANGE_ID = os.getenv("EXCHANGE","bitvavo").strip().lower()

CAPITAL_EUR       = float(os.getenv("CAPITAL_EUR","1100"))
BAND_PCT          = float(os.getenv("BAND_PCT","0.20"))
COINS_CSV         = os.getenv("COINS","BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR")
DATA_DIR_STR      = os.getenv("DATA_DIR","data")
FEE_PCT           = float(os.getenv("FEE_PCT","0.0015"))
GRID_LEVELS       = int(os.getenv("GRID_LEVELS","32"))
LOCK_PREEXISTING  = _b("LOCK_PREEXISTING_BALANCE","true")
LOG_SUMMARY_SEC   = int(os.getenv("LOG_SUMMARY_SEC","240"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR","25"))
MIN_PROFIT_EUR    = float(os.getenv("MIN_PROFIT_EUR","0.10"))
MIN_PROFIT_PCT    = float(os.getenv("MIN_PROFIT_PCT","0.001"))
MIN_QUOTE_EUR     = float(os.getenv("MIN_QUOTE_EUR","5"))
OPERATOR_ID       = os.getenv("OPERATOR_ID","").strip()
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR","1.6"))
REINVEST_PROFITS  = _b("REINVEST_PROFITS","true")
REPORT_EVERY_HOURS= float(os.getenv("REPORT_EVERY_HOURS","4"))
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC","300"))
SLEEP_SEC         = int(os.getenv("SLEEP_SEC","3"))
WEIGHTS_CSV       = os.getenv("WEIGHTS","BTC/EUR:0.45,ETH/EUR:0.20,SOL/EUR:0.25,XRP/EUR:0.05,LTC/EUR:0.05")
SELL_SAFETY_PCT   = float(os.getenv("SELL_SAFETY_PCT","0.003"))  # 0.3% default
STOP_LOSS_PCT     = float(os.getenv("STOP_LOSS_PCT","0"))        # 0 = uit

# base-min overrides "PAIR:amount"
def parse_overrides(s: str):
    out={}
    for it in [x.strip() for x in os.getenv("MIN_BASE_OVERRIDES","").split(",") if x.strip()]:
        if ":" in it:
            k,v=it.split(":",1)
            try: out[k.strip().upper()] = float(v)
            except: pass
    # XRP/EUR heeft vrijwel altijd 2.0 min base
    out.setdefault("XRP/EUR", 2.0)
    return out
MIN_BASE_OVERRIDES = parse_overrides(os.getenv("MIN_BASE_OVERRIDES",""))

# ========= storage =========
try:
    DATA_DIR = Path(DATA_DIR_STR); DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE  = DATA_DIR/"live_state.json"
TRADES_CSV  = DATA_DIR/"live_trades.csv"
EQUITY_CSV  = DATA_DIR/"live_equity.csv"

# ========= exchange =========
def make_exchange():
    if not API_KEY or not API_SECRET:
        raise SystemExit("API_KEY / API_SECRET ontbreken.")
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    if OPERATOR_ID: ex.options["operatorId"] = OPERATOR_ID
    ex.load_markets()
    return ex

def free_eur_on_exchange(ex)->float:
    try: return float((ex.fetch_balance().get("free") or {}).get("EUR") or 0.0)
    except Exception: return 0.0

def free_base_on_exchange(ex, base)->float:
    try: return float((ex.fetch_balance().get("free") or {}).get(base) or 0.0)
    except Exception: return 0.0

def amount_to_precision(ex, pair, qty: float)->float:
    try: return float(ex.amount_to_precision(pair, qty))
    except Exception: return qty

def best_bid_px(ex, pair) -> float:
    try:
        ob = ex.fetch_order_book(pair, limit=5)
        if ob and ob.get("bids"): return float(ob["bids"][0][0])
    except Exception: pass
    t = ex.fetch_ticker(pair)
    return float(t.get("bid") or t.get("last"))

# ========= grid support =========
def normalize_weights(pairs, weights_csv):
    d={}
    for it in [x.strip() for x in weights_csv.split(",") if x.strip()]:
        if ":" in it:
            k,v=it.split(":",1)
            try: d[k.strip().upper()] = float(v)
            except: pass
    s = sum(d.values())
    return ({p: 1.0/len(pairs) for p in pairs} if s<=0 else {p:(d.get(p,0.0)/s) for p in pairs})

def geometric_levels(low, high, n):
    if low<=0 or high<=0 or n<2: return [low, high]
    r = (high/low)**(1/(n-1))
    return [low*(r**i) for i in range(n)]

def compute_band_from_history(ex, pair):
    try:
        o = ex.fetch_ohlcv(pair, timeframe="15m", limit=4*24*14)
        if o and len(o)>=100:
            s = pd.Series([c[4] for c in o if c and c[4] is not None])
            p10=float(s.quantile(0.10)); p90=float(s.quantile(0.90))
            if p90>p10>0: return p10, p90
    except Exception: pass
    last=float(ex.fetch_ticker(pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def mk_grid_state(ex, pair, levels):
    low,high=compute_band_from_history(ex,pair)
    return {"pair":pair,"low":low,"high":high,"levels":geometric_levels(low,high,levels),
            "last_price":None,"inventory_lots":[]}

def init_portfolio(pairs, weights):
    return {
        "play_cash_eur": CAPITAL_EUR,
        "cash_eur": CAPITAL_EUR,
        "pnl_realized": 0.0,
        "coins": {p: {"qty":0.0, "cash_alloc": CAPITAL_EUR*weights[p]} for p in pairs}
    }

def euro_per_ticket(cash_alloc, n_levels):
    if n_levels<2: n_levels=2
    base = (cash_alloc*0.90)/(n_levels//2)
    return max(5.0, base*ORDER_SIZE_FACTOR)

def invested_cost_eur(state):
    tot=0.0
    for g in state["grids"].values():
        for l in g["inventory_lots"]: tot += l["qty"]*l["buy_price"]
    return tot

def mark_to_market(ex, state, pairs):
    total = state["portfolio"]["cash_eur"]
    for p in pairs:
        q = state["portfolio"]["coins"][p]["qty"]
        if q>0:
            px = float(ex.fetch_ticker(p)["last"])
            total += q*px
    return total

def cap_now(state)->float:
    pnl=float(state.get("portfolio",{}).get("pnl_realized",0.0))
    return CAPITAL_EUR if REINVEST_PROFITS else max(0.0, min(CAPITAL_EUR, CAPITAL_EUR - pnl))

# ====== market minima ======
def market_mins(ex, pair, price_now: float):
    m = ex.markets.get(pair,{}) or {}
    lim = m.get("limits") or {}
    min_cost=float((lim.get("cost") or {}).get("min") or 0) or 0.0
    min_amt =float((lim.get("amount") or {}).get("min") or 0) or 0.0
    info = m.get("info") or {}
    mc = float(info.get("minOrderInQuoteAsset") or 0) or 0.0
    ma = float(info.get("minOrderInBaseAsset") or 0) or 0.0
    min_amt  = max(min_amt, ma, MIN_BASE_OVERRIDES.get(pair, 0.0))
    min_cost = max(min_cost, mc, MIN_QUOTE_EUR)
    if min_amt<=0 and price_now>0: min_amt = (min_cost/price_now)*1.02
    return float(min_cost), float(min_amt)

# ===== TP / SL =====
def trigger_price(buy_price: float) -> float:
    """Verkoopdrempel (TP) = winst + 2×fee + safety."""
    return buy_price * (1.0 + MIN_PROFIT_PCT + 2.0*FEE_PCT + SELL_SAFETY_PCT)

def stop_price(buy_price: float) -> float:
    """Stop-loss prijs (0 = uit)."""
    if STOP_LOSS_PCT <= 0: return 0.0
    return buy_price * (1.0 - STOP_LOSS_PCT)

# ===== cost-cap aware BUY/SELL =====
def buy_market(ex, pair, eur):
    """Market buy met Bitvavo cost-param. eur is al <= beschikbare EUR (incl. buffer) en >= minima."""
    if eur < MIN_QUOTE_EUR: return 0.0,0.0,0.0,0.0
    params={"cost": float(f"{eur:.2f}")}
    if OPERATOR_ID: params["operatorId"]=OPERATOR_ID
    o = ex.create_order(pair, "market", "buy", None, None, params)
    avg   = float(o.get("average") or o.get("price") or 0.0)
    filled= float(o.get("filled") or o.get("info",{}).get("filledAmount") or 0.0)
    executed = avg*filled
    fee = executed*FEE_PCT
    return filled, avg, fee, executed

def sell_market(ex, pair, qty):
    if qty<=0: return 0.0,0.0,0.0
    params={}
    if OPERATOR_ID: params["operatorId"]=OPERATOR_ID
    o = ex.create_order(pair, "market", "sell", qty, None, params)
    avg = float(o.get("average") or o.get("price") or 0.0)
    proceeds = avg*qty
    fee = proceeds*FEE_PCT
    return proceeds, avg, fee

def net_gain_ok(buy_price, sell_avg, qty):
    if buy_price<=0 or sell_avg<=0 or qty<=0: return False
    net_eur = (sell_avg - buy_price)*qty - (sell_avg*qty + buy_price*qty)*FEE_PCT
    net_pct = (sell_avg/buy_price) - 1.0 - 2.0*FEE_PCT
    return (net_pct>=MIN_PROFIT_PCT) or (net_eur>=MIN_PROFIT_EUR)

def bot_inventory_value_eur_from_exchange(ex, state, pairs)->float:
    try: free = (ex.fetch_balance().get("free") or {})
    except Exception: free={}
    baselines = state.get("baseline", {}) if LOCK_PREEXISTING else {}
    tot=0.0
    for p in pairs:
        base=p.split("/")[0]
        qty=max(0.0, float(free.get(base) or 0.0) - float(baselines.get(base,0.0) or 0.0))
        if qty>0:
            px=float(ex.fetch_ticker(p)["last"]); tot += qty*px
    return tot

# ========= core =========
def try_grid_live(ex, pair, price_now, price_prev, state, grid, pairs):
    logs=[]
    levels=grid["levels"]; port=state["portfolio"]
    if not levels: return logs

    # BUY branch — prijs daalt door een level
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now < L <= price_prev]
        if crossed:
            avail = free_eur_on_exchange(ex)
            live_cap = bot_inventory_value_eur_from_exchange(ex, state, pairs)
            cap = cap_now(state)
        for _ in crossed:
            ticket = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels))

            min_quote, min_base = market_mins(ex, pair, price_now)
            required = max(min_quote, min_base*price_now)

            # maximaal wat we mogen/moeten kunnen spenderen
            max_cost_by_cash = max(0.0, avail - MIN_CASH_BUFFER_EUR)/(1.0 + FEE_PCT)
            # aangescherpt: nooit boven ticket
            cost_plan = min(ticket, max_cost_by_cash)

            if cost_plan < required - 1e-6:
                logs.append(f"{COL_C}[{pair}] BUY skip: free_EUR=€{avail:.2f} < need≈€{required:.2f} (incl. buffer).{COL_RESET}")
                continue

            # rond naar boven op centen maar niet boven cash-limiet
            eur = min(cost_plan, math.floor(max_cost_by_cash*100)/100)
            eur = max(eur, required)
            eur = math.ceil(eur*100)/100  # altijd >= minima

            # kostenplafonds
            if invested_cost_eur(state)+eur > cap + 1e-9:
                logs.append(f"{COL_C}[{pair}] BUY skip: cost-cap bereikt (cap=€{cap:.2f}).{COL_RESET}"); continue
            if live_cap+eur > cap + 1e-9:
                logs.append(f"{COL_C}[{pair}] BUY skip: live-cap (≈€{live_cap:.2f}/{cap:.2f}).{COL_RESET}"); continue

            # voer uit
            qty, avg, fee, executed = buy_market(ex, pair, eur)
            if qty<=0 or avg<=0:
                logs.append(f"{COL_C}[{pair}] BUY fill < minima of rejected; overslaan.{COL_RESET}")
                continue

            # Controleer minima op fill
            if qty < min_base or executed < min_quote:
                logs.append(f"{COL_C}[{pair}] BUY fill < minima (amt={qty:.8f}/{min_base}, eur={executed:.2f}/{min_quote}); overslaan.{COL_RESET}")
                continue

            grid["inventory_lots"].append({"qty":qty, "buy_price":avg})
            port["cash_eur"] -= (executed + fee)
            port["coins"][pair]["qty"] += qty
            avail -= (executed + fee); live_cap += executed

            tp = trigger_price(avg)
            append_csv(TRADES_CSV,
                [now_iso(),pair,"BUY",f"{avg:.6f}",f"{qty:.8f}",f"{executed:.2f}",f"{port['cash_eur']:.2f}",
                 pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
                header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"])
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{avg:.6f} | ticket≈€{eur:.2f} | fee≈€{fee:.2f} | TP=€{tp:.6f} | cash=€{port['cash_eur']:.2f}")

    # SELL branch — TP of SL
    if grid["inventory_lots"]:
        base = pair.split("/")[0]
        bot_free = None
        if LOCK_PREEXISTING and "baseline" in state:
            bot_free = max(0.0, free_base_on_exchange(ex, base) - float(state["baseline"].get(base,0.0)))

        min_quote, min_base = market_mins(ex, pair, price_now)
        changed=True
        while changed and grid["inventory_lots"]:
            changed=False
            # kies eerste lot dat TP of SL raakt
            idx=None; reason="TP"
            for i,l in enumerate(grid["inventory_lots"]):
                tp = trigger_price(l["buy_price"])
                sl = stop_price(l["buy_price"])
                bid = best_bid_px(ex, pair)
                if STOP_LOSS_PCT>0 and bid<=sl and bid>0:
                    idx=i; reason="SL"; break
                if bid>=tp: idx=i; reason="TP"; break
            if idx is None:
                # wachtregel voor het oudste lot
                l = grid["inventory_lots"][0]
                bid = best_bid_px(ex, pair); trig = trigger_price(l["buy_price"])
                logs.append(f"[{pair}] SELL wait: bid €{bid:.6f} < trigger €{trig:.6f} (buy €{l['buy_price']:.6f})")
                break

            lot = grid["inventory_lots"][idx]
            qty = lot["qty"]

            if qty < min_base or (qty*price_now) < min_quote:
                logs.append(f"[{pair}] SELL skip: lot te klein (amt {qty:.8f}/{min_base} of €{qty*price_now:.2f}/€{min_quote:.2f}).")
                break

            if bot_free is not None and bot_free + 1e-12 < qty:
                logs.append(f"[{pair}] SELL stop: baseline protect ({bot_free:.8f} {base} beschikbaar).")
                break

            sell_qty = amount_to_precision(ex, pair, qty)
            proceeds, avg, fee = sell_market(ex, pair, sell_qty)
            if proceeds>0 and avg>0:
                ok = (reason=="SL") or net_gain_ok(lot["buy_price"], avg, sell_qty)
                if not ok:
                    # theoretisch zou dit niet moeten gebeuren (bid check), maar be safe
                    break
                grid["inventory_lots"].pop(idx)
                pnl = proceeds - fee - (sell_qty*lot["buy_price"])
                port["cash_eur"] += (proceeds - fee)
                port["coins"][pair]["qty"] -= sell_qty
                port["pnl_realized"] += pnl
                if bot_free is not None: bot_free -= sell_qty

                append_csv(TRADES_CSV,
                    [now_iso(),pair,"SELL",f"{avg:.6f}",f"{sell_qty:.8f}",f"{proceeds:.2f}",f"{port['cash_eur']:.2f}",
                     base, f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", ("stop_loss" if reason=="SL" else "take_profit")])
                col = COL_G if pnl>=0 else COL_R
                logs.append(f"{col}[{pair}] SELL {sell_qty:.8f} @ €{avg:.6f} (buy €{lot['buy_price']:.6f}) | proceeds=€{proceeds:.2f} | fee=€{fee:.2f} | pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}")
                changed=True

    grid["last_price"]=price_now
    return logs

# ========= main =========
def main():
    ex = make_exchange()
    pairs=[x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    pairs=[p for p in pairs if p in ex.markets]
    if not pairs: raise SystemExit("Geen geldige markten gevonden.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)
    state = load_json(STATE_FILE,{})
    if "portfolio" not in state: state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state: state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]: state["grids"][p] = mk_grid_state(ex,p,GRID_LEVELS)

    if LOCK_PREEXISTING and "baseline" not in state:
        bal = ex.fetch_balance().get("free",{})
        state["baseline"] = {p.split("/")[0]: float(bal.get(p.split("/")[0],0) or 0.0) for p in pairs}

    save_json(STATE_FILE, state)

    print(f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={pct(FEE_PCT)} | "
          f"pairs={pairs} | factor={ORDER_SIZE_FACTOR} | min_profit={pct(MIN_PROFIT_PCT)} / €{MIN_PROFIT_EUR:.2f} | "
          f"sell_safety={pct(SELL_SAFETY_PCT)} | stop_loss={(pct(STOP_LOSS_PCT) if STOP_LOSS_PCT>0 else 'uit')}")

    last_sum = 0.0; last_report = 0.0

    while True:
        try:
            eq = mark_to_market(ex, state, pairs)
            append_csv(EQUITY_CSV, [datetime.now(timezone.utc).date().isoformat(), f"{eq:.2f}"],
                       header=["date","total_equity_eur"])

            if time.time()-last_sum >= LOG_SUMMARY_SEC:
                since=eq-CAPITAL_EUR; col=COL_G if since>=0 else COL_R
                pr=state["portfolio"]["pnl_realized"]; cash=state["portfolio"]["cash_eur"]
                inv=invested_cost_eur(state); fe=free_eur_on_exchange(ex)
                live=bot_inventory_value_eur_from_exchange(ex,state,pairs); cap=cap_now(state)
                print(f"[SUMMARY] total_eq=€{eq:.2f} | cash=€{cash:.2f} | free_EUR=€{fe:.2f} | invested_cost=€{inv:.2f} | "
                      f"live_inv=€{live:.2f} | cap_now=€{cap:.2f} | pnl_realized=€{pr:.2f} | since_start={col}{since:.2f}{COL_RESET}")
                last_sum=time.time()

            for p in pairs:
                px = float(ex.fetch_ticker(p)["last"])
                logs = try_grid_live(ex, p, px, state["grids"][p]["last_price"], state, state["grids"][p], pairs)
                if logs: print("\n".join(logs))

            if time.time()-last_report >= REPORT_EVERY_HOURS*3600:
                pr=state["portfolio"]["pnl_realized"]; cash=state["portfolio"]["cash_eur"]
                inv=invested_cost_eur(state); fe=free_eur_on_exchange(ex)
                live=bot_inventory_value_eur_from_exchange(ex,state,pairs); cap=cap_now(state)
                eq = mark_to_market(ex,state,pairs); since=eq-CAPITAL_EUR; col=COL_G if since>=0 else COL_R
                print(f"[REPORT] total_eq=€{eq:.2f} | cash=€{cash:.2f} | free_EUR=€{fe:.2f} | invested_cost=€{inv:.2f} | "
                      f"live_inv=€{live:.2f} | cap_now=€{cap:.2f} | pnl_realized=€{pr:.2f} | since_start={col}{since:.2f}{COL_RESET} | pairs={pairs}")
                last_report=time.time()

            save_json(STATE_FILE, state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[net] {e}; backoff.."); time.sleep(2+random.random())
        except ccxt.BaseError as e:
            print(f"[ccxt] {e}; wacht.."); time.sleep(5)
        except KeyboardInterrupt:
            print("Gestopt."); break
        except Exception as e:
            print(f"[runtime] {e}"); time.sleep(5)

if __name__=="__main__":
    main()
