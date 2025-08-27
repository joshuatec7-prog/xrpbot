# ==============================
#  gridbot main.py  (ETH/EUR)
#  - Verkoopt nooit onder break-even + min_profit (incl. fees), behalve bij STOP_LOSS
#  - Trailing take-profit
#  - One-position-only + cooldowns
#  - Bitvavo (ccxt) live-only
# ==============================

import os, time, csv, pathlib, logging, random, math
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
import ccxt
import pandas as pd

# ---------- helpers ----------
def now_iso():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def _mask(s):
    if not s: return "MISSING"
    return s[:4] + "..." + s[-4:]

# ---------- ENV (alfabetisch) ----------
API_KEY  = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.5"))
COOLDOWN_SEC = int(os.getenv("COOLDOWN_SEC", "180"))
EUR_PER_TRADE = float(os.getenv("EUR_PER_TRADE", "15"))
FAST = int(os.getenv("FAST", "3"))
FEE_PCT = float(os.getenv("FEE_PCT", "0.0015"))  # 0.15%/order
LIVE = os.getenv("LIVE", "true").lower() in ("1","true","yes","y")
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE", "false").lower() in ("1","true","yes","y")
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC", "600"))
LOG_TRADES = os.getenv("LOG_TRADES", "true").lower() in ("1","true","yes","y")
MAX_BUYS_PER_POSITION = int(os.getenv("MAX_BUYS_PER_POSITION", "1"))
MAX_EUR_PER_TRADE = float(os.getenv("MAX_EUR_PER_TRADE", "20"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.005"))  # 0.5%
ONE_POSITION_ONLY = os.getenv("ONE_POSITION_ONLY", "true").lower() in ("1","true","yes","y")
OPERATOR_ID = os.getenv("OPERATOR_ID", "12234").strip()
ORDER_ID_PREFIX = os.getenv("ORDER_ID_PREFIX", "gridbot")
REQUEST_INTERVAL_SEC = float(os.getenv("REQUEST_INTERVAL_SEC", "1.0"))
SLEEP_SEC = int(os.getenv("SLEEP_SEC", "60"))
SLOW = int(os.getenv("SLOW", "6"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.02"))    # 2% max verlies
SYMBOL = os.getenv("SYMBOL", "ETH/EUR").upper()
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.02"))  # vaste TP 2%
TRADE_COOLDOWN_SEC = int(os.getenv("TRADE_COOLDOWN_SEC", "180"))
TRAIL_GAP = float(os.getenv("TRAIL_GAP", "0.005"))     # 0.5% onder piek
TRAIL_START = float(os.getenv("TRAIL_START", "0.015")) # start lock bij +1.5%
TRAIL_TRIGGER = float(os.getenv("TRAIL_TRIGGER", "0.02"))  # activeer bij +2%

if not LIVE:
    raise SystemExit("Deze bot draait alleen LIVE. Zet LIVE=true.")

if not API_KEY or not API_SECRET:
    raise SystemExit("GEEN API_SLEUTELS: zet API_KEY en API_SECRET in Secrets.")

# ---------- Logging ----------
LOG_DIR = pathlib.Path(".")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("gridbot")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = RotatingFileHandler(LOG_DIR / "bot.log", maxBytes=512_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

TRADES_CSV = LOG_DIR / "trades.csv"
def log_trade(ts, side, price, amount, reason, pnl_eur=0.0):
    if not LOG_TRADES:
        return
    new = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp","symbol","side","price","amount","reason","pnl_eur"])
        w.writerow([ts, SYMBOL, side, f"{price:.8f}", f"{amount:.8f}", reason, f"{pnl_eur:.2f}"])

# ---------- Exchange ----------
ex = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
})
ex.options["createMarketBuyOrderRequiresPrice"] = False  # Bitvavo buy via "cost"

markets = ex.load_markets()
if SYMBOL not in markets:
    raise SystemExit(f"Symbool {SYMBOL} niet gevonden op Bitvavo.")

BASE, QUOTE = SYMBOL.split("/")
_limits = markets[SYMBOL].get("limits", {})
MIN_BASE  = float((_limits.get("amount") or {}).get("min") or 0.0)
MIN_QUOTE = float((_limits.get("cost")   or {}).get("min") or 0.0)

def amount_to_precision(symbol, amount: float) -> float:
    return float(ex.amount_to_precision(symbol, amount))

# ---------- retry / basic pacing ----------
_last_call = 0.0
def _respect_interval():
    global _last_call
    wait = REQUEST_INTERVAL_SEC - (time.time() - _last_call)
    if wait > 0: time.sleep(wait)
    _last_call = time.time()

def with_retry(fn, *a, **kw):
    delay = 1.0
    for i in range(1, MAX_RETRIES+1):
        try:
            _respect_interval()
            return fn(*a, **kw)
        except (ccxt.NetworkError, ccxt.RateLimitExceeded, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            sleep_s = delay + random.uniform(0, 0.5)
            print(f"[retry {i}/{MAX_RETRIES}] {e} -> sleep {sleep_s:.2f}s")
            logger.warning("retry %s/%s: %s", i, MAX_RETRIES, e)
            time.sleep(sleep_s)
            delay *= BACKOFF_BASE
        except Exception as e:
            logger.exception("unexpected: %s", e)
            time.sleep(1.0)
    raise RuntimeError("Max retries reached")

# ---------- signaal ----------
def fetch_df(limit=SLOW+5) -> pd.DataFrame:
    o = with_retry(ex.fetch_ohlcv, SYMBOL, timeframe="1m", limit=limit)
    return pd.DataFrame(o, columns=["t","o","h","l","c","v"])

def signal(df: pd.DataFrame) -> str:
    if len(df) < max(FAST, SLOW)+3: return "HOLD"
    f = df["c"].rolling(FAST).mean()
    s = df["c"].rolling(SLOW).mean()
    # gebruik de voorlaatste candle voor cross
    if f.iloc[-2] > s.iloc[-2] and f.iloc[-3] <= s.iloc[-3]:
        return "BUY"
    if f.iloc[-2] < s.iloc[-2] and f.iloc[-3] >= s.iloc[-3]:
        return "SELL"
    return "HOLD"

# ---------- balans lock ----------
start_bal = with_retry(ex.fetch_balance).get("free", {})
start_eur  = float(start_bal.get(QUOTE, 0) or 0)
start_base = float(start_bal.get(BASE, 0)  or 0)

def bot_balances():
    bal = with_retry(ex.fetch_balance).get("free", {})
    eur  = float(bal.get(QUOTE, 0) or 0)
    base = float(bal.get(BASE, 0) or 0)
    if LOCK_PREEXISTING_BALANCE:
        eur_for_bot  = max(0.0, eur - start_eur)
        base_for_bot = max(0.0, base - start_base)
    else:
        eur_for_bot, base_for_bot = eur, base
    return eur_for_bot, base_for_bot

# ---------- positie / trailing state ----------
entry_price = None
position_amt = 0.0
buys_this_position = 0

trail_active = False
trail_peak = None
trail_floor = None

def reset_trailing():
    global trail_active, trail_peak, trail_floor
    trail_active = False
    trail_peak = None
    trail_floor = None

last_trade_ts = 0.0
last_summary_ts = 0.0

# ---------- prijsdrempels ----------
def break_even_sell_price(entry: float) -> float:
    # koop-fee + verkoop-fee + minimaal gewenste winst
    # (1 + FEE + MIN_PROFIT + FEE) = 1 + 2*FEE + MIN_PROFIT
    return entry * (1.0 + (2.0 * FEE_PCT) + MIN_PROFIT_PCT)

def fixed_take_profit(entry: float) -> float:
    return entry * (1.0 + TAKE_PROFIT)

def stop_loss_price(entry: float) -> float:
    return entry * (1.0 - STOP_LOSS)

# ---------- orders ----------
def place_market_buy(eur_spend, px_now):
    params = {"cost": eur_spend}
    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
    with_retry(ex.create_order, SYMBOL, "market", "buy", None, None, params)
    logger.info("BUY ~€%.2f @ €%.6f", eur_spend, px_now)

def place_market_sell(amount, px_now):
    params = {}
    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
    with_retry(ex.create_order, SYMBOL, "market", "sell", amount, None, params)
    logger.info("SELL %.8f @ €%.6f", amount, px_now)

# ---------- main ----------
print(f"START gridbot | {SYMBOL} | LIVE={LIVE} | €/trade={EUR_PER_TRADE} | fee={FEE_PCT*100:.3f}%")
logger.info("START symbol=%s live=%s eur_per_trade=%.2f", SYMBOL, LIVE, EUR_PER_TRADE)

while True:
    try:
        df = fetch_df()
        px = float(df["c"].iloc[-1])
        sig = signal(df)

        eur_free, base_free = bot_balances()

        # periodic summary
        if time.time() - last_summary_ts >= LOG_SUMMARY_SEC:
            print(f"[SUMMARY] cash=€{eur_free:.2f} pos={position_amt:.6f} {BASE} entry={entry_price if entry_price else 0:.4f}")
            last_summary_ts = time.time()

        # ----- trailing / SL / TP voor bestaande positie -----
        if position_amt > 0 and entry_price:
            chg = (px - entry_price) / entry_price

            # trailing activeren bij trigger
            if not trail_active and chg >= TRAIL_TRIGGER:
                trail_active = True
                trail_peak = px
                trail_floor = max(entry_price * (1.0 + TRAIL_START), trail_peak * (1.0 - TRAIL_GAP))
                print(f"[TRAIL] Activated peak=€{trail_peak:.6f} floor=€{trail_floor:.6f}")

            if trail_active:
                if px > trail_peak:
                    trail_peak = px
                    trail_floor = trail_peak * (1.0 - TRAIL_GAP)
                elif px <= trail_floor:
                    # trailing sell – check min-amount / min-cost
                    to_sell = min(position_amt, base_free)
                    to_sell = amount_to_precision(SYMBOL, to_sell)
                    if to_sell >= MIN_BASE and to_sell*px >= max(MIN_QUOTE, 0.50):
                        place_market_sell(to_sell, px)
                        pnl = (px - entry_price) * to_sell - (px*to_sell*FEE_PCT) - (entry_price*to_sell*FEE_PCT)
                        log_trade(now_iso(), "SELL", px, to_sell, "TRAIL", pnl)
                        position_amt -= to_sell
                        if position_amt <= 1e-9:
                            position_amt = 0.0
                            entry_price = None
                            buys_this_position = 0
                            reset_trailing()
                        last_trade_ts = time.time()
                        time.sleep(SLEEP_SEC)
                        continue

            # vaste TP of SL
            if px >= fixed_take_profit(entry_price) or px <= stop_loss_price(entry_price):
                reason = "TAKE_PROFIT" if px >= fixed_take_profit(entry_price) else "STOP_LOSS"
                to_sell = min(position_amt, base_free)
                to_sell = amount_to_precision(SYMBOL, to_sell)
                if to_sell >= MIN_BASE and to_sell*px >= max(MIN_QUOTE, 0.50):
                    place_market_sell(to_sell, px)
                    pnl = (px - entry_price) * to_sell - (px*to_sell*FEE_PCT) - (entry_price*to_sell*FEE_PCT)
                    log_trade(now_iso(), "SELL", px, to_sell, reason, pnl)
                    position_amt -= to_sell
                    if position_amt <= 1e-9:
                        position_amt = 0.0
                        entry_price = None
                        buys_this_position = 0
                        reset_trailing()
                    last_trade_ts = time.time()
                    time.sleep(SLEEP_SEC)
                    continue

        # ----- BUY -----
        if sig == "BUY":
            # cooldown & one-position rules
            if time.time() - last_trade_ts < COOLDOWN_SEC:
                pass
            elif ONE_POSITION_ONLY and position_amt > 0 and buys_this_position >= MAX_BUYS_PER_POSITION:
                pass
            else:
                eur_room = eur_free
                if eur_room > 0:
                    target = min(EUR_PER_TRADE, MAX_EUR_PER_TRADE, eur_room)
                    # respecteer minimumeisen
                    need_eur = max(MIN_QUOTE, (MIN_BASE * px))
                    if target + 1e-9 < need_eur:
                        # probeer naar boven bij te stellen binnen je cap/room
                        target = min(max(target, need_eur), eur_room, MAX_EUR_PER_TRADE)
                    spend = math.floor(target*100)/100  # 2 decimalen EUR
                    est_base = spend / px
                    if spend >= max(MIN_QUOTE, 0.5) and est_base >= MIN_BASE:
                        place_market_buy(spend, px)
                        # conservatief: neem gevulde qty ≈ spend/px
                        got = est_base
                        position_amt += got
                        # gemiddelde entry als er al iets ligt
                        if entry_price:
                            entry_price = ((entry_price * (position_amt-got)) + (px * got)) / position_amt
                        else:
                            entry_price = px
                        buys_this_position += 1
                        log_trade(now_iso(), "BUY", px, got, "SIGNAL")
                        reset_trailing()
                        last_trade_ts = time.time()

        # ----- SELL (signaal) -> alleen als winst-drempel gehaald -----
        elif sig == "SELL" and position_amt > 1e-9 and entry_price:
            # alleen verkopen als prijs boven break-even + min_profit (inclusief fees)
            sell_ok_price = max(break_even_sell_price(entry_price), fixed_take_profit(entry_price)*0.9999)
            if px >= sell_ok_price or px <= stop_loss_price(entry_price):
                to_sell = min(position_amt, base_free)
                to_sell = amount_to_precision(SYMBOL, to_sell)
                if to_sell >= MIN_BASE and to_sell*px >= max(MIN_QUOTE, 0.50):
                    place_market_sell(to_sell, px)
                    pnl = (px - entry_price) * to_sell - (px*to_sell*FEE_PCT) - (entry_price*to_sell*FEE_PCT)
                    reason = "SIGNAL"
                    if px <= stop_loss_price(entry_price):
                        reason = "STOP_LOSS"
                    log_trade(now_iso(), "SELL", px, to_sell, reason, pnl)
                    position_amt -= to_sell
                    if position_amt <= 1e-9:
                        position_amt = 0.0
                        entry_price = None
                        buys_this_position = 0
                        reset_trailing()
                    last_trade_ts = time.time()

        time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Gestopt door gebruiker.")
        logger.info("STOP by user")
        break
    except Exception as e:
        print("[runtime] error:", e)
        logger.exception("runtime error: %s", e)
        time.sleep(5)

