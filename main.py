# main.py
# LIVE GRID/MA BOT (Bitvavo) met trailing stops, SL/TP, anti double-buy,
# logt naar trades.csv en print nette regels in Render logs.

import os, time, csv, pathlib, random, json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import pandas as pd
import ccxt

from keep_alive import keep_alive
keep_alive()  # klein webservertje

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):  # alles naar Render logs
    print(f"[{now_iso()}] {msg}")

# ------------------ ENV ------------------
SYMBOL   = os.getenv("SYMBOL", "ETH/EUR").upper()
TIMEFRAME= os.getenv("TIMEFRAME", "1m")
FAST     = int(os.getenv("FAST", "3"))
SLOW     = int(os.getenv("SLOW", "6"))

LIVE     = os.getenv("LIVE", "true").lower() in ("1","true","yes","y")
EUR_PER_TRADE     = float(os.getenv("EUR_PER_TRADE", "15"))
MAX_EUR_PER_TRADE = float(os.getenv("MAX_EUR_PER_TRADE", "20"))

STOP_LOSS   = float(os.getenv("STOP_LOSS", "0.01"))
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.03"))
TRAIL_TRIGGER = float(os.getenv("TRAIL_TRIGGER", "0.02"))
TRAIL_START   = float(os.getenv("TRAIL_START", "0.015"))
TRAIL_GAP     = float(os.getenv("TRAIL_GAP", "0.01"))

LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE", "false").lower() in ("1","true","yes","y")

# anti-dubbel
ONE_POSITION_ONLY     = os.getenv("ONE_POSITION_ONLY","true").lower() in ("1","true","yes","y")
MAX_BUYS_PER_POSITION = int(os.getenv("MAX_BUYS_PER_POSITION", "1"))
TRADE_COOLDOWN_SEC    = int(os.getenv("TRADE_COOLDOWN_SEC", "180"))
COOLDOWN_SEC          = int(os.getenv("COOLDOWN_SEC", "120"))
ORDER_ID_PREFIX       = os.getenv("ORDER_ID_PREFIX", "gridbot")
OPERATOR_ID           = os.getenv("OPERATOR_ID","").strip()

SLEEP_SEC     = float(os.getenv("SLEEP_SEC","60"))
MAX_RETRIES   = int(os.getenv("MAX_RETRIES","5"))
BACKOFF_BASE  = float(os.getenv("BACKOFF_BASE","1.5"))
REQ_INTERVAL  = float(os.getenv("REQUEST_INTERVAL_SEC","1.0"))

API_KEY    = os.getenv("API_KEY","")
API_SECRET = os.getenv("API_SECRET","")

if not API_KEY or not API_SECRET:
    raise SystemExit("GEEN API KEYS (API_KEY/API_SECRET)")

LOG_TRADES      = os.getenv("LOG_TRADES","true").lower() in ("1","true","yes","y")
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC","600"))

DATA_DIR   = pathlib.Path(".")
TRADES_CSV = DATA_DIR / "trades.csv"

def append_trade(row):
    new = not TRADES_CSV.exists()
    with open(TRADES_CSV,"a",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp","symbol","side","price","amount_or_cost","fee","cash_eur_after","reason"])
        w.writerow(row)

# ------------------ Exchange ------------------
ex = ccxt.bitvavo({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
ex.options["createMarketBuyOrderRequiresPrice"] = False
markets = ex.load_markets()
if SYMBOL not in markets:
    raise SystemExit(f"Symbool {SYMBOL} niet op Bitvavo.")

BASE, QUOTE = SYMBOL.split("/")
_limits = markets[SYMBOL].get("limits", {})
MIN_BASE  = float((_limits.get("amount") or {}).get("min") or 0.0)
MIN_QUOTE = float((_limits.get("cost")   or {}).get("min") or 0.0)
log(f"Minima {SYMBOL}: min_cost=€{MIN_QUOTE}, min_amount={MIN_BASE} {BASE}")

def amount_to_precision(symbol: str, amount: float) -> float:
    return float(ex.amount_to_precision(symbol, amount))

# rate limit helper
_last_call = 0.0
def throttle():
    global _last_call
    wait = REQ_INTERVAL - (time.time() - _last_call)
    if wait > 0: time.sleep(wait)
    _last_call = time.time()

def with_retry(call, *args, **kwargs):
    delay = 1.0
    for i in range(1, MAX_RETRIES+1):
        try:
            throttle()
            return call(*args, **kwargs)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            time.sleep(delay)
            delay *= BACKOFF_BASE
        except Exception as e:
            time.sleep(1.0)
    raise RuntimeError("Max retries exceeded")

# ------------------ Signaal ------------------
def fetch_df(limit=SLOW+5) -> pd.DataFrame:
    o = with_retry(ex.fetch_ohlcv, SYMBOL, timeframe=TIMEFRAME, limit=limit)
    return pd.DataFrame(o, columns=["t","o","h","l","c","v"])

def signal(df: pd.DataFrame) -> str:
    if len(df) < max(FAST,SLOW)+2:
        return "HOLD"
    f = df["c"].rolling(FAST).mean()
    s = df["c"].rolling(SLOW).mean()
    if f.iloc[-2] > s.iloc[-2] and f.iloc[-3] <= s.iloc[-3]: return "BUY"
    if f.iloc[-2] < s.iloc[-2] and f.iloc[-3] >= s.iloc[-3]: return "SELL"
    return "HOLD"

# ------------------ State ------------------
state = {
    "start_bal": with_retry(ex.fetch_balance).get("free",{}),
    "position_amt": 0.0,
    "entry_price": None,
    "trail_active": False, "trail_peak": None, "trail_floor": None,
    "last_trade_ts": 0.0,
    "buys_this_position": 0,
}
start_eur  = float(state["start_bal"].get(QUOTE,0) or 0)
start_base = float(state["start_bal"].get(BASE,0)  or 0)

def avail_for_bot() -> Dict[str,float]:
    bal = with_retry(ex.fetch_balance).get("free",{})
    eur = float(bal.get(QUOTE,0) or 0)
    base= float(bal.get(BASE ,0) or 0)
    if LOCK_PREEXISTING_BALANCE:
        return {"eur":max(0.0, eur-start_eur), "base":max(0.0, base-start_base)}
    return {"eur":eur, "base":base}

def reset_trailing():
    state["trail_active"]=False; state["trail_peak"]=None; state["trail_floor"]=None

def maybe_trailing(px: float):
    if state["entry_price"] is None or state["position_amt"]<=0: return
    chg = (px - state["entry_price"]) / state["entry_price"]

    if not state["trail_active"] and chg >= TRAIL_TRIGGER:
        state["trail_active"]=True
        state["trail_peak"]=px
        state["trail_floor"]=max(state["entry_price"]*(1+TRAIL_START), px*(1-TRAIL_GAP))
        log(f"[TRAIL] Activated peak=€{state['trail_peak']:.6f} floor=€{state['trail_floor']:.6f}")

    if state["trail_active"]:
        if px > state["trail_peak"]:
            state["trail_peak"]=px
            state["trail_floor"]=state["trail_peak"]*(1-TRAIL_GAP)
        if px <= state["trail_floor"]:
            # verkoop alles binnen limieten
            sell_sz = min(state["position_amt"], avail_for_bot()["base"])
            sell_sz = amount_to_precision(SYMBOL, sell_sz)
            if sell_sz >= MIN_BASE and (sell_sz*px) >= max(MIN_QUOTE,0.5):
                params={}
                if OPERATOR_ID: params["operatorId"]=OPERATOR_ID
                with_retry(ex.create_order, SYMBOL, "market", "sell", sell_sz, None, params)
                state["position_amt"] -= sell_sz
                append_trade([now_iso(),SYMBOL,"SELL",f"{px:.6f}",f"{sell_sz:.8f}","-",f"{avail_for_bot()['eur']:.2f}","TRAILING_STOP"])
                if LOG_TRADES: log(f"{SYMBOL} SELL {sell_sz:.8f} @ €{px:.6f} (trailing)")
                if state["position_amt"]<=0: state["entry_price"]=None; reset_trailing()

def do_buy(px: float):
    # cooldown
    if time.time() - state["last_trade_ts"] < TRADE_COOLDOWN_SEC:
        return
    if ONE_POSITION_ONLY and state["position_amt"]>0:
        return
    if state["buys_this_position"] >= MAX_BUYS_PER_POSITION:
        return

    avail = avail_for_bot()["eur"]
    target = min(EUR_PER_TRADE, MAX_EUR_PER_TRADE, avail)
    need_min = max(MIN_QUOTE, MIN_BASE*px, 0.50)
    if target < need_min:
        return

    spend = round(target,2)
    params = {"cost": spend}
    if OPERATOR_ID: params["operatorId"]=OPERATOR_ID
    order = with_retry(ex.create_order, SYMBOL, "market", "buy", None, None, params)

    try:
        filled = float(order.get("info",{}).get("filledAmount") or 0.0)
    except Exception:
        filled = spend/px
    state["position_amt"] += filled
    if state["entry_price"] is None: state["entry_price"]=px
    state["last_trade_ts"] = time.time()
    state["buys_this_position"] += 1
    append_trade([now_iso(),SYMBOL,"BUY",f"{px:.6f}",f"€{spend:.2f}","-",f"{avail_for_bot()['eur']:.2f}","SIGNAL"])
    if LOG_TRADES: log(f"{SYMBOL} BUY ~€{spend:.2f} (+{filled:.6f}) @ €{px:.6f}")

def do_sell(px: float, reason: str):
    if state["position_amt"]<=0: return
    avail = avail_for_bot()["base"]
    sell_sz = min(state["position_amt"], avail, MAX_EUR_PER_TRADE/px)
    sell_sz = amount_to_precision(SYMBOL, sell_sz)
    if sell_sz < MIN_BASE or (sell_sz*px) < max(MIN_QUOTE,0.5):
        return
    params={}
    if OPERATOR_ID: params["operatorId"]=OPERATOR_ID
    with_retry(ex.create_order, SYMBOL, "market", "sell", sell_sz, None, params)
    state["position_amt"] -= sell_sz
    if state["position_amt"]<=0:
        state["entry_price"]=None
        state["buys_this_position"]=0
        reset_trailing()
    state["last_trade_ts"] = time.time()
    append_trade([now_iso(),SYMBOL,"SELL",f"{px:.6f}",f"{sell_sz:.8f}","-",f"{avail_for_bot()['eur']:.2f}",reason])
    if LOG_TRADES: log(f"{SYMBOL} SELL {sell_sz:.8f} @ €{px:.6f} ({reason})")

# ------------------ Main loop ------------------
log(f"START | {SYMBOL} | TF={TIMEFRAME} | live={LIVE} | €/trade={EUR_PER_TRADE} | SL={STOP_LOSS*100:.1f}% | TP={TAKE_PROFIT*100:.1f}%")
last_summary = 0.0

while True:
    try:
        df = fetch_df()
        px = float(df["c"].iloc[-1])
        sig = signal(df)
        if LOG_TRADES: log(f"Prijs=€{px:.4f} | Signaal={sig}")

        # trailing + SL/TP
        maybe_trailing(px)
        if state["entry_price"]:
            chg = (px - state["entry_price"]) / state["entry_price"]
            if chg >= TAKE_PROFIT:
                do_sell(px, "TAKE_PROFIT")
            elif chg <= -STOP_LOSS:
                do_sell(px, "STOP_LOSS")

        # signaal-acties
        if sig == "BUY":
            do_buy(px)
        elif sig == "SELL":
            do_sell(px, "SIGNAL")

        # periodieke samenvatting
        if LOG_SUMMARY_SEC > 0 and time.time() - last_summary >= LOG_SUMMARY_SEC:
            a = avail_for_bot()
            log(f"SUMMARY cash=€{a['eur']:.2f} pos={state['position_amt']:.6f} {BASE} entry={state['entry_price'] if state['entry_price'] else '-'}")
            last_summary = time.time()

        time.sleep(SLEEP_SEC)
    except KeyboardInterrupt:
        break
    except Exception as e:
        log(f"[runtime] {e}")
        time.sleep(5)
