# futures_grid.py — Kraken Futures grid met LONG + SHORT
# - Exchange: krakenfutures (ccxt)
# - Symbols automatisch: <BASE>/USD:USD swap (linear)
# - Koopt LONG op neerwaartse cross, opent SHORT op opwaartse cross
# - Sluit lots bij netto winst + safety (LONG: sell-close, SHORT: buy-close)
# - DRY_RUN=true voor paper; false voor live
# - CSV: data/fut_trades.csv, data/fut_equity.csv

import os, csv, json, time, random, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import ccxt
import pandas as pd

# ==== ANSI ====
COL_G="\033[92m"; COL_R="\033[91m"; COL_C="\033[96m"; COL_RESET="\033[0m"

def now_iso(): return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
def pct(x): return f"{x*100:.2f}%"

def append_csv(path: Path, row, header=None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
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

# ==== ENV ====
API_KEY=os.getenv("API_KEY","")
API_SECRET=os.getenv("API_SECRET","")
EXCHANGE_ID=os.getenv("EXCHANGE","krakenfutures").strip().lower()

BAND_PCT=float(os.getenv("BAND_PCT","0.15"))
CAPITAL_USD=float(os.getenv("CAPITAL_USD","10000"))
COINS=[x.strip().upper() for x in os.getenv("COINS","BTC,ETH,SOL").split(",") if x.strip()]
DATA_DIR=Path(os.getenv("DATA_DIR","data"))
DRY_RUN=os.getenv("DRY_RUN","true").lower() in ("1","true","yes")
FEE_PCT=float(os.getenv("FEE_PCT","0.0005"))
GRID_LEVELS=int(os.getenv("GRID_LEVELS","48"))
LEVERAGE=float(os.getenv("LEVERAGE","5"))
LOG_SUMMARY_SEC=int(os.getenv("LOG_SUMMARY_SEC","240"))
MAX_NOTIONAL_PER_SIDE_USD=float(os.getenv("MAX_NOTIONAL_PER_SIDE_USD","2500"))
MIN_CASH_BUFFER_USD=float(os.getenv("MIN_CASH_BUFFER_USD","200"))
MIN_PROFIT_PCT=float(os.getenv("MIN_PROFIT_PCT","0.003"))
MIN_QUOTE_USD=float(os.getenv("MIN_QUOTE_USD","5"))
OPERATOR_ID=os.getenv("OPERATOR_ID","").strip()
ORDER_SIZE_FACTOR=float(os.getenv("ORDER_SIZE_FACTOR","1.2"))
REPORT_EVERY_HOURS=float(os.getenv("REPORT_EVERY_HOURS","4"))
SELL_SAFETY_PCT=float(os.getenv("SELL_SAFETY_PCT","0.006"))
SLEEP_HEARTBEAT_SEC=int(os.getenv("SLEEP_HEARTBEAT_SEC","300"))
SLEEP_SEC=int(os.getenv("SLEEP_SEC","3"))
WEIGHTS_CSV=os.getenv("WEIGHTS","BTC:0.4,ETH:0.25,SOL:0.2,XRP:0.1,LINK:0.05").strip()

STATE_FILE=DATA_DIR/"fut_state.json"
TRADES_CSV=DATA_DIR/"fut_trades.csv"
EQUITY_CSV=DATA_DIR/"fut_equity.csv"

if not API_KEY or not API_SECRET:
    print("[warn] API_KEY/API_SECRET ontbreken; DRY_RUN wordt geforceerd")
    DRY_RUN=True

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==== Exchange ====
def make_exchange():
    klass=getattr(ccxt, EXCHANGE_ID)
    ex=klass({"apiKey":API_KEY,"secret":API_SECRET,"enableRateLimit":True})
    if OPERATOR_ID:
        ex.options["operatorId"]=OPERATOR_ID
    ex.load_markets()
    return ex

def amount_to_precision(ex, symbol, qty: float)->float:
    try: return float(ex.amount_to_precision(symbol, qty))
    except Exception: return qty

def price_to_precision(ex, symbol, px: float)->float:
    try: return float(ex.price_to_precision(symbol, px))
    except Exception: return px

# ==== Markets ====
def resolve_usd_swap(ex, base: str) -> Optional[str]:
    # zoek linear USD swaps voor base
    for sid, m in ex.markets.items():
        try:
            if m.get("type")=="swap" and m.get("linear") and m.get("base")==base and m.get("quote")=="USD":
                return m["symbol"]   # bv "BTC/USD:USD"
        except Exception:
            continue
    return None

# ==== Weights ====
def normalize_weights(bases: List[str], weights_csv: str)->Dict[str,float]:
    d={}
    for it in [x.strip() for x in weights_csv.split(",") if x.strip()]:
        if ":" in it:
            k,v=it.split(":",1)
            try: d[k.strip().upper()]=float(v)
            except: pass
    s=sum(d.values()) or 0.0
    return ({b:1.0/len(bases) for b in bases} if s<=0 else {b:(d.get(b,0.0)/s) for b in bases})

# ==== Grid ====
def geometric_levels(low, high, n):
    if low<=0 or high<=0 or n<2: return [low,high]
    r=(high/low)**(1/(n-1))
    return [low*(r**i) for i in range(n)]

def compute_band_from_history(ex, symbol: str) -> Tuple[float,float]:
    try:
        o=ex.fetch_ohlcv(symbol, timeframe="15m", limit=4*24*14)
        if o and len(o)>=100:
            closes=[c[4] for c in o if c and c[4] is not None]
            s=pd.Series(closes); p10=float(s.quantile(0.10)); p90=float(s.quantile(0.90))
            if p90>p10>0: return p10,p90
    except Exception:
        pass
    last=float(ex.fetch_ticker(symbol)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def mk_grid_state(ex, symbol: str, levels: int)->dict:
    low,high=compute_band_from_history(ex, symbol)
    return {"symbol":symbol,"low":low,"high":high,"levels":geometric_levels(low,high,levels),
            "last_price":None,"long_lots":[],"short_lots":[]}

# ==== Portfolio / sizing ====
def init_portfolio(bases: List[str], weights: Dict[str,float])->dict:
    return {
        "cash_usd": CAPTIAL_PLAY(),
        "pnl_realized": 0.0,
        "bases": {b: {"notional_alloc": CAPTIAL_PLAY()*weights[b], "long_base":0.0, "short_base":0.0} for b in bases}
    }

def CAPTIAL_PLAY():
    return CAPTIAL_NOW()

def CAPTIAL_NOW():
    return CAPITAL_USD

def ticket_usd(alloc: float, n_levels: int)->float:
    if n_levels<2: n_levels=2
    base=(alloc*0.90)/(n_levels//2)
    return max(MIN_QUOTE_USD, base*ORDER_SIZE_FACTOR)

# ==== PnL helpers ====
def net_gain_ok(entry: float, exit_px: float, fee_pct: float, min_pct: float, qty_base: float, side: str)->bool:
    if entry<=0 or exit_px<=0 or qty_base<=0: return False
    gross = (exit_px-entry)/entry if side=="long" else (entry-exit_px)/entry
    net_pct = gross - 2.0*fee_pct
    return net_pct >= min_pct

# ==== Orders (live/paper) ====
def create_market_order(ex, symbol: str, side: str, qty_base: float):
    # side: "buy" or "sell"
    if DRY_RUN:
        t=ex.fetch_ticker(symbol); px=float(t["last"])
        avg=px; filled=qty_base
        return {"side":side,"filled":filled,"average":avg,"price":avg,"cost":avg*filled}
    else:
        order = ex.create_order(symbol, "market", side, qty_base, None, {})
        return order

def best_bid(ex, symbol: str)->float:
    try:
        ob=ex.fetch_order_book(symbol, limit=5)
        if ob and ob.get("bids"): return float(ob["bids"][0][0])
    except Exception:
        pass
    t=ex.fetch_ticker(symbol)
    return float(t.get("bid") or t.get("last"))

def best_ask(ex, symbol: str)->float:
    try:
        ob=ex.fetch_order_book(symbol, limit=5)
        if ob and ob.get("asks"): return float(ob["asks"][0][0])
    except Exception:
        pass
    t=ex.fetch_ticker(symbol)
    return float(t.get("ask") or t.get("last"))

# ==== Core per symbol ====
def try_futures_grid(ex, base: str, state: dict, grid: dict, weights: Dict[str,float], caps: Dict[str,float])->List[str]:
    logs=[]
    symbol=grid["symbol"]; levels=grid["levels"]; if not levels: return logs
    t=ex.fetch_ticker(symbol); px=float(t["last"])
    prev=grid["last_price"]
    alloc = state["portfolio"]["bases"][base]["notional_alloc"]

    # sizing per lot
    lot_usd = ticket_usd(alloc, len(levels))
    lot_base = lot_usd / px

    # guard: max notional per side
    def side_notional(side: str) -> float:
        return sum([(l["qty_base"]*px) for l in (grid["long_lots"] if side=="long" else grid["short_lots"])])

    # BUY LONG op neerwaartse cross
    if prev is not None and px < prev:
        crossed=[L for L in levels if px < L <= prev]
        for _ in crossed:
            if side_notional("long")+lot_usd > caps["per_side"]:
                logs.append(f"{COL_C}[{base}] LONG skip: cap {side_notional('long'):.2f}/{caps['per_side']:.2f}{COL_RESET}")
                continue
            qty = amount_to_precision(ex, symbol, lot_base)
            if qty*px < MIN_QUOTE_USD: continue
            # market BUY (open/increase long)
            o=create_market_order(ex, symbol, "buy", qty)
            avg=float(o.get("average") or o.get("price") or px)
            fee=(avg*qty)*FEE_PCT
            grid["long_lots"].append({"qty_base":qty,"entry":avg})
            append_csv(
                TRADES_CSV,
                [now_iso(), symbol, "LONG_OPEN", f"{avg:.6f}", f"{qty:.8f}", f"{avg*qty:.2f}", "", "", "", "", "grid_long_open"],
                header=["timestamp","symbol","side","price","qty_base","notional_usd","pnl_usd","dir","comment","unused1","unused2"]
            )
            logs.append(f"{COL_C}[{base}] LONG +{qty:.6f} @ ${avg:.2f}{COL_RESET}")

    # OPEN SHORT op opwaartse cross
    if prev is not None and px > prev:
        crossed=[L for L in levels if prev < L <= px]
        for _ in crossed:
            if side_notional("short")+lot_usd > caps["per_side"]:
                logs.append(f"{COL_C}[{base}] SHORT skip: cap {side_notional('short'):.2f}/{caps['per_side']:.2f}{COL_RESET}")
                continue
            qty = amount_to_precision(ex, symbol, lot_base)
            if qty*px < MIN_QUOTE_USD: continue
            # market SELL (open/increase short)
            o=create_market_order(ex, symbol, "sell", qty)
            avg=float(o.get("average") or o.get("price") or px)
            fee=(avg*qty)*FEE_PCT
            grid["short_lots"].append({"qty_base":qty,"entry":avg})
            append_csv(TRADES_CSV,[now_iso(), symbol, "SHORT_OPEN", f"{avg:.6f}", f"{qty:.8f}", f"{avg*qty:.2f}", "", "", "", "", "grid_short_open"])
            logs.append(f"{COL_C}[{base}] SHORT +{qty:.6f} @ ${avg:.2f}{COL_RESET}")

    # TAKE PROFIT LONGS (verkoop tegen best bid)
    if grid["long_lots"]:
        bid=best_bid(ex, symbol)
        sold_any=True
        while sold_any and grid["long_lots"]:
            sold_any=False
            idx=next((i for i,l in enumerate(grid["long_lots"])
                      if net_gain_ok(l["entry"], bid, FEE_PCT, MIN_PROFIT_PCT, l["qty_base"], "long")
                      and bid >= l["entry"]*(1+MIN_PROFIT_PCT+2*FEE_PCT+SELL_SAFETY_PCT)), None)
            if idx is None: break
            lot=grid["long_lots"].pop(idx)
            qty=amount_to_precision(ex, symbol, lot["qty_base"])
            o=create_market_order(ex, symbol, "sell", qty)
            avg=float(o.get("average") or o.get("price") or bid)
            pnl=(avg - lot["entry"])*qty - (avg*qty)*FEE_PCT - (lot["entry"]*qty)*FEE_PCT
            state["portfolio"]["pnl_realized"] += pnl
            append_csv(TRADES_CSV,[now_iso(), symbol, "LONG_CLOSE", f"{avg:.6f}", f"{qty:.8f}", f"{avg*qty:.2f}", f"{pnl:.2f}", "long", "tp", "", ""])
            logs.append(f"{(COL_G if pnl>=0 else COL_R)}[{base}] LONG_CLOSE {qty:.6f} pnl=${pnl:.2f}{COL_RESET}")
            sold_any=True

    # TAKE PROFIT SHORTS (koop terug tegen best ask)
    if grid["short_lots"]:
        ask=best_ask(ex, symbol)
        bought_any=True
        while bought_any and grid["short_lots"]:
            bought_any=False
            idx=next((i for i,l in enumerate(grid["short_lots"])
                      if net_gain_ok(l["entry"], ask, FEE_PCT, MIN_PROFIT_PCT, l["qty_base"], "short")
                      and ask <= l["entry"]*(1-MIN_PROFIT_PCT-2*FEE_PCT-SELL_SAFETY_PCT)), None)
            if idx is None: break
            lot=grid["short_lots"].pop(idx)
            qty=amount_to_precision(ex, symbol, lot["qty_base"])
            o=create_market_order(ex, symbol, "buy", qty)
            avg=float(o.get("average") or o.get("price") or ask)
            pnl=(lot["entry"] - avg)*qty - (avg*qty)*FEE_PCT - (lot["entry"]*qty)*FEE_PCT
            state["portfolio"]["pnl_realized"] += pnl
            append_csv(TRADES_CSV,[now_iso(), symbol, "SHORT_CLOSE", f"{avg:.6f}", f"{qty:.8f}", f"{avg*qty:.2f}", f"{pnl:.2f}", "short", "tp", "", ""])
            logs.append(f"{(COL_G if pnl>=0 else COL_R)}[{base}] SHORT_CLOSE {qty:.6f} pnl=${pnl:.2f}{COL_RESET}")
            bought_any=True

    grid["last_price"]=px
    return logs

# ==== Equity (ruwe benadering) ====
def mark_to_market_usd(ex, state: dict, grids: Dict[str,dict])->float:
    eq = CAPTIAL_NOW()
    for base, g in grids.items():
        if not g["last_price"]: continue
        px=g["last_price"]
        # long-lots waarde
        for l in g["long_lots"]:
            eq += l["qty_base"] * px
        # short-lots reserveren geen cash; ongerealiseerde wordt niet meegeteld
    return eq

# ==== Main ====
def main():
    ex=make_exchange()
    # symbol mapping
    base2symbol: Dict[str,str] = {}
    for b in COINS:
        sym=resolve_usd_swap(ex, b)
        if sym: base2symbol[b]=sym
    if not base2symbol:
        raise SystemExit("Geen geschikte USD swaps gevonden")

    weights=normalize_weights(list(base2symbol.keys()), WEIGHTS_CSV)

    state = load_json(STATE_FILE, {}) if STATE_FILE.exists() else {}
    if "portfolio" not in state:
        state["portfolio"]={"cash_usd": CAPTIAL_NOW(), "pnl_realized": 0.0, "bases": {b: {"notional_alloc": CAPTIAL_NOW()*weights[b]} for b in base2symbol.keys()}}
    if "grids" not in state: state["grids"]={}
    for b, sym in base2symbol.items():
        if b not in state["grids"]:
            state["grids"][b]=mk_grid_state(ex, sym, GRID_LEVELS)

    save_json(STATE_FILE, state)

    print(f"== FUTURES GRID start | dry_run={DRY_RUN} | capital=${CAPTIAL_NOW():.2f} | fee={pct(FEE_PCT)} | "
          f"levels={GRID_LEVELS} | min_profit={pct(MIN_PROFIT_PCT)} | safety={pct(SELL_SAFETY_PCT)} | pairs={base2symbol}")

    last_report=0.0; last_sum=0.0
    caps={"per_side": MAX_NOTIONAL_PER_SIDE_USD}

    while True:
        try:
            eq=mark_to_market_usd(ex, state, state["grids"])
            append_csv(EQUITY_CSV, [datetime.now(timezone.utc).date().isoformat(), f"{eq:.2f}"], header=["date","equity_usd"])

            now_ts=time.time()
            if now_ts - last_sum >= LOG_SUMMARY_SEC:
                pr=state["portfolio"]["pnl_realized"]
                print(f"[SUMMARY] equity≈${eq:.2f} | pnl_realized=${pr:.2f}")
                last_sum=now_ts

            for b in list(base2symbol.keys()):
                logs=try_futures_grid(ex, b, state, state["grids"][b], weights, caps)
                if logs: print("\n".join(logs))

            if time.time()-last_report >= REPORT_EVERY_HOURS*3600:
                pr=state["portfolio"]["pnl_realized"]
                print(f"[REPORT] equity≈${eq:.2f} | pnl_realized=${pr:.2f} | bases={list(base2symbol.keys())}")
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
