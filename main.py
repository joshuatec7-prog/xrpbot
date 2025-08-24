# ==============================
#  LIVE grid-achtige swing-bot (ccxt + Bitvavo)
#  - Render Background Worker (geen webserver)
#  - 1 positie tegelijk (configurable)
#  - Koop-cap + cooldowns
#  - Minima-checks (min cost / min amount)
#  - Trailing/TP/SL optioneel
# ==============================

import os, time, csv, pathlib, logging, random, math
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
import pandas as pd
import ccxt

# ---------- helpers ----------
def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

def _mask(s: str | None) -> str:
    if not s: return "MISSING"
    return s[:4] + "..." + s[-4:]

# ---------- Config (via ENV) ----------
LIVE   = os.environ.get("LIVE", "true").lower() in ("1","true","yes","y")
SYMBOL = os.environ.get("SYMBOL", "ETH/EUR")
TIMEFRAME = os.environ.get("TIMEFRAME", "1m")

FAST = int(os.environ.get("FAST", "3"))
SLOW = int(os.environ.get("SLOW", "6"))

# Kooplimieten
EUR_PER_TRADE     = float(os.environ.get("EUR_PER_TRADE", "15"))   # jouw inzet
MAX_EUR_PER_TRADE = float(os.environ.get("MAX_EUR_PER_TRADE", "20"))  # harde cap

# Positie-regels
ONE_POSITION_ONLY     = os.environ.get("ONE_POSITION_ONLY", "true").lower() in ("1","true","yes","y")
MAX_BUYS_PER_POSITION = int(os.environ.get("MAX_BUYS_PER_POSITION", "1"))

# Cooldowns
COOLDOWN_SEC        = int(os.environ.get("COOLDOWN_SEC", "75"))     # tussen opeenvolgende BUY-signalen
TRADE_COOLDOWN_SEC  = int(os.environ.get("TRADE_COOLDOWN_SEC", "180"))  # na een daadwerkelijke order

# (optioneel) trailing/TP/SL
TRAIL_TRIGGER = float(os.environ.get("TRAIL_TRIGGER", "0.02"))  # 2%
TRAIL_START   = float(os.environ.get("TRAIL_START", "0.015"))   # 1.5%
TRAIL_GAP     = float(os.environ.get("TRAIL_GAP", "0.01"))      # 1%
TAKE_PROFIT   = float(os.environ.get("TAKE_PROFIT", "0.03"))    # 3%
STOP_LOSS     = float(os.environ.get("STOP_LOSS", "0.01"))      # 1%

LOCK_PREEXISTING_BALANCE = os.environ.get("LOCK_PREEXISTING_BALANCE", "false").lower() in ("1","true","yes","y")

SLEEP_SEC = int(os.environ.get("SLEEP_SEC", "60"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "1.5"))
REQ_INTERVAL = float(os.environ.get("REQUEST_INTERVAL_SEC", "1.0"))

API_KEY    = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
OPERATOR_ID = os.environ.get("OPERATOR_ID", "").strip()  # Bitvavo account-operator id (optioneel)
ORDER_ID_PREFIX = os.environ.get("ORDER_ID_PREFIX", "gridbot")

if not LIVE:
    raise SystemExit("Deze worker is LIVE-only. Zet LIVE=true in Environment Variables.")
if not API_KEY or not API_SECRET:
    raise SystemExit("GEEN API SLEUTELS: zet API_KEY en API_SECRET in Environment Variables.")

print("API_KEY    =", _mask(API_KEY))
print("API_SECRET =", _mask(API_SECRET))
print("SYMBOL     =", SYMBOL)

# ---------- logging ----------
LOG_DIR = pathlib.Path(".")
LOG_DIR.mkdir(exist_ok=True)
bot_logger = logging.getLogger("bot")
bot_logger.setLevel(logging.INFO)
if not bot_logger.handlers:
    h = RotatingFileHandler(LOG_DIR / "bot.log", maxBytes=512_000, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    bot_logger.addHandler(h)

TRADES_CSV = LOG_DIR / "trades.csv"
def log_trade(ts_iso: str, side: str, price: float, amount_or_cost: float, reason: str):
    new = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(["timestamp","side","price","qty_or_cost","reason"])
        w.writerow([ts_iso, side, f"{price:.8f}", f"{amount_or_cost:.8f}", reason])

# ---------- Exchange ----------
ex = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
})
ex.options["createMarketBuyOrderRequiresPrice"] = False  # we gebruiken 'cost' voor market buy

markets = ex.load_markets()
if SYMBOL not in markets:
    raise SystemExit(f"Symbool {SYMBOL} niet gevonden op Bitvavo.")
BASE, QUOTE = SYMBOL.split("/")
_limits = markets[SYMBOL].get("limits", {})
MIN_BASE  = float((_limits.get("amount") or {}).get("min") or 0.0)
MIN_QUOTE = float((_limits.get("cost")   or {}).get("min") or 0.0)
print(f"[INFO] Minima {SYMBOL}: min_cost=€{MIN_QUOTE}, min_amount={MIN_BASE} {BASE}")

def amt_to_precision(symbol: str, amount: float) -> float:
    return float(ex.amount_to_precision(symbol, amount))

# ---------- Retry + pacing ----------
_last_call_ts = 0.0
def _respect_min_interval():
    global _last_call_ts
    now = time.time()
    wait = REQ_INTERVAL - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()

def with_retry(call, *args, **kwargs):
    delay = 1.0
    for attempt in range(1, MAX_RETRIES+1):
        try:
            _respect_min_interval()
            return call(*args, **kwargs)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            bot_logger.warning("retry %s/%s: %s", attempt, MAX_RETRIES, e)
            time.sleep(delay + random.uniform(0, 0.5))
            delay *= BACKOFF_BASE
        except Exception as e:
            bot_logger.exception("unexpected: %s", e)
            time.sleep(1.0)
    raise RuntimeError("Max retries reached for exchange call.")

# ---------- Data & Signaal ----------
def fetch_ohlcv(limit=SLOW + 3) -> pd.DataFrame:
    ohlcv = with_retry(ex.fetch_ohlcv, SYMBOL, timeframe=TIMEFRAME, limit=limit)
    return pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])

def ma_signal(df: pd.DataFrame) -> str:
    if len(df) < max(FAST, SLOW) + 2: return "HOLD"
    f = df["c"].rolling(FAST).mean()
    s = df["c"].rolling(SLOW).mean()
    cross_up   = f.iloc[-2] > s.iloc[-2] and f.iloc[-3] <= s.iloc[-3]
    cross_down = f.iloc[-2] < s.iloc[-2] and f.iloc[-3] >= s.iloc[-3]
    if cross_up: return "BUY"
    if cross_down: return "SELL"
    return "HOLD"

# ---------- startbalans (lock) ----------
bal0 = with_retry(ex.fetch_balance).get("free", {})
start_eur  = float(bal0.get(QUOTE, 0) or 0)
start_base = float(bal0.get(BASE, 0)  or 0)

# ---------- trade state ----------
position_open = False
buys_in_position = 0
position_amt = 0.0
entry_price = None

trail_active = False
trail_peak = None
trail_floor = None

last_signal_ts = 0.0     # cooldown op BUY-signaal
last_trade_ts  = 0.0     # cooldown na daadwerkelijke order
EPS = 1e-10

def reset_trailing():
    global trail_active, trail_peak, trail_floor
    trail_active = False
    trail_peak = None
    trail_floor = None

def reset_position():
    global position_open, buys_in_position, position_amt, entry_price
    position_open = False
    buys_in_position = 0
    position_amt = 0.0
    entry_price = None
    reset_trailing()

def free_balances_for_bot():
    bal = with_retry(ex.fetch_balance).get("free", {})
    eur_free  = float(bal.get(QUOTE, 0) or 0)
    base_free = float(bal.get(BASE, 0) or 0)
    if LOCK_PREEXISTING_BALANCE:
        return max(0.0, eur_free - start_eur), max(0.0, base_free - start_base)
    return eur_free, base_free

print(
    f"START | {SYMBOL} | TF={TIMEFRAME} | €/trade={EUR_PER_TRADE} | CAP={MAX_EUR_PER_TRADE} | "
    f"one_pos={ONE_POSITION_ONLY} | max_buys={MAX_BUYS_PER_POSITION}"
)
bot_logger.info("START | symbol=%s tf=%s live=%s", SYMBOL, TIMEFRAME, LIVE)

# ---------- hoofdloop ----------
while True:
    try:
        df = fetch_ohlcv()
        px = float(df["c"].iloc[-1])
        sig = ma_signal(df)

        eur_for_bot, base_for_bot = free_balances_for_bot()
        print(f"Prijs: €{px:.4f} | Signaal: {sig}")
        print(f"[DEBUG] Vrij {QUOTE}(bot): {eur_for_bot:.2f} | Vrij {BASE}(bot): {base_for_bot:.6f}")

        now = time.time()

        # trailing + SL/TP
        if position_open and position_amt > EPS and entry_price:
            change = (px - entry_price) / entry_price

            # trailing activeren
            if not trail_active and change >= TRAIL_TRIGGER:
                trail_active = True
                trail_peak = px
                trail_floor = max(entry_price * (1 + TRAIL_START), trail_peak * (1 - TRAIL_GAP))
                print(f"[TRAIL] Activated | peak €{trail_peak:.6f} | floor €{trail_floor:.6f}")

            if trail_active:
                if px > trail_peak:
                    trail_peak = px
                    trail_floor = trail_peak * (1 - TRAIL_GAP)
                if px <= trail_floor:
                    # SELL via trailing
                    to_sell = min(position_amt, base_for_bot)
                    to_sell = amt_to_precision(SYMBOL, to_sell)
                    if to_sell >= max(MIN_BASE, 1e-12) and to_sell * px >= max(MIN_QUOTE, 0.50):
                        params = {}
                        if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
                        params["clientOrderId"] = f"{ORDER_ID_PREFIX}-trail-{int(time.time())}"
                        with_retry(ex.create_order, SYMBOL, "market", "sell", to_sell, None, params)
                        print(f"[TRAIL] SELL {to_sell} {BASE} @ €{px:.6f}")
                        bot_logger.info("TRAIL SELL %.8f @ %.6f", to_sell, px)
                        log_trade(now_iso(), "SELL", px, to_sell, "TRAIL")
                        position_amt = max(0.0, position_amt - to_sell)
                        if position_amt <= EPS:
                            reset_position()
                            last_trade_ts = now
                    else:
                        print("[TRAIL] Te klein -> overslaan")

            # backstop TP/SL
            if position_open and (change >= TAKE_PROFIT or change <= -STOP_LOSS):
                reason = "TP" if change >= TAKE_PROFIT else "SL"
                to_sell = min(position_amt, base_for_bot)
                to_sell = amt_to_precision(SYMBOL, to_sell)
                if to_sell >= max(MIN_BASE, 1e-12) and to_sell * px >= max(MIN_QUOTE, 0.50):
                    params = {}
                    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
                    params["clientOrderId"] = f"{ORDER_ID_PREFIX}-{reason.lower()}-{int(time.time())}"
                    with_retry(ex.create_order, SYMBOL, "market", "sell", to_sell, None, params)
                    print(f"[LIVE] {reason} SELL {to_sell} {BASE} @ €{px:.6f}")
                    bot_logger.info("%s SELL %.8f @ %.6f", reason, to_sell, px)
                    log_trade(now_iso(), "SELL", px, to_sell, reason)
                    position_amt = max(0.0, position_amt - to_sell)
                    if position_amt <= EPS:
                        reset_position()
                        last_trade_ts = now
                else:
                    print("[LIVE] SL/TP te klein -> overslaan")

        # ----------------- kooplogica -----------------
        if sig == "BUY":
            # cooldowns
            if now - last_signal_ts < COOLDOWN_SEC:
                print("[COOLDOWN] BUY-signaal genegeerd")
            elif now - last_trade_ts < TRADE_COOLDOWN_SEC:
                print("[COOLDOWN] Na trade, even wachten")
            elif ONE_POSITION_ONLY and position_open and buys_in_position >= MAX_BUYS_PER_POSITION:
                print("[RULE] Al in positie (max buys bereikt)")
            else:
                # check beschikbare EUR
                if eur_for_bot <= 0:
                    print(f"[LIVE] Geen {QUOTE} beschikbaar boven startbalans.")
                else:
                    target = min(EUR_PER_TRADE, MAX_EUR_PER_TRADE, eur_for_bot)
                    # min vereist
                    min_required = max(MIN_QUOTE, (MIN_BASE * px if MIN_BASE > 0 else 0.0), 0.50)
                    if target + 1e-9 < min_required:
                        print(f"[LIVE] BUY overslaan: nodig ≥ €{min_required:.2f}")
                    else:
                        spend = round(target, 2)
                        # market buy via cost
                        params = {"cost": spend}
                        if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
                        params["clientOrderId"] = f"{ORDER_ID_PREFIX}-buy-{int(time.time())}"
                        order = with_retry(ex.create_order, SYMBOL, "market", "buy", None, None, params)
                        # gevuld
                        filled = None
                        try:
                            filled = float(order.get("info", {}).get("filledAmount"))
                        except Exception:
                            pass
                        if not filled:
                            filled = spend / px
                        position_amt += filled
                        entry_price = px if not position_open else (entry_price or px)
                        position_open = True
                        buys_in_position += 1
                        reset_trailing()

                        print(f"[LIVE] BUY ~€{spend:.2f} -> +{filled:.6f} {BASE} @ €{px:.6f}")
                        bot_logger.info("BUY cost=%.2f, filled=%.8f @ %.6f", spend, filled, px)
                        log_trade(now_iso(), "BUY", px, spend, "SIGNAL")
                        last_trade_ts = now
            last_signal_ts = now

        # ----------------- verkoop op SELL-signaal -----------------
        elif sig == "SELL":
            if position_open and position_amt > EPS and base_for_bot > 0:
                # verkoop maximaal CAP per stap
                max_base_for_cap = MAX_EUR_PER_TRADE / px
                to_sell = min(position_amt, base_for_bot, max_base_for_cap)
                to_sell = amt_to_precision(SYMBOL, to_sell)
                if to_sell >= max(MIN_BASE, 1e-12) and to_sell * px >= max(MIN_QUOTE, 0.50):
                    params = {}
                    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
                    params["clientOrderId"] = f"{ORDER_ID_PREFIX}-sell-{int(time.time())}"
                    with_retry(ex.create_order, SYMBOL, "market", "sell", to_sell, None, params)
                    print(f"[LIVE] SIGNAL SELL {to_sell} {BASE} @ €{px:.6f}")
                    bot_logger.info("SIGNAL SELL %.8f @ %.6f", to_sell, px)
                    log_trade(now_iso(), "SELL", px, to_sell, "SIGNAL")
                    position_amt = max(0.0, position_amt - to_sell)
                    if position_amt <= EPS:
                        reset_position()
                        last_trade_ts = now
                else:
                    print("[LIVE] SELL te klein -> overslaan")
            else:
                print("[LIVE] Niets (van deze sessie) om te verkopen.")

        time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Gestopt door gebruiker.")
        bot_logger.info("STOP by user")
        break
    except Exception as e:
        print("[LIVE] ❌ Runtime error:", repr(e))
        bot_logger.exception("Runtime error: %s", e)
        time.sleep(5)
