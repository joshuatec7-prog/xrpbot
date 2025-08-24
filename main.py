# ==============================
#  mini-bot (Render/Bitvavo) – safe & idempotent
# ==============================
import os, time, csv, pathlib, logging, random, uuid
from logging.handlers import RotatingFileHandler
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import ccxt

# ---------- kleine helpers ----------
def now_iso() -> str:
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%d-%m-%Y %H:%M:%S")

def _mask(s: str | None) -> str:
    if not s:
        return "MISSING"
    return s[:4] + "..." + s[-4:]

# ---------- ENV / instellingen ----------
SYMBOL    = os.environ.get("SYMBOL", "XRP/EUR")
TIMEFRAME = os.environ.get("TIMEFRAME", "1m")
FAST      = int(os.environ.get("FAST", "3"))
SLOW      = int(os.environ.get("SLOW", "6"))
LIVE      = os.environ.get("LIVE", "true").lower() in ("1","true","yes","y")

EUR_PER_TRADE     = float(os.environ.get("EUR_PER_TRADE", "7.50"))
MAX_EUR_PER_TRADE = float(os.environ.get("MAX_EUR_PER_TRADE", "10"))

# Risk / trailing
STOP_LOSS     = float(os.environ.get("STOP_LOSS", "0.01"))      # 1%
TAKE_PROFIT   = float(os.environ.get("TAKE_PROFIT", "0.03"))    # 3%
TRAIL_TRIGGER = float(os.environ.get("TRAIL_TRIGGER", "0.02"))  # +2%
TRAIL_START   = float(os.environ.get("TRAIL_START", "0.015"))   # +1.5%
TRAIL_GAP     = float(os.environ.get("TRAIL_GAP", "0.01"))      # 1%

LOCK_PREEXISTING_BALANCE = os.environ.get(
    "LOCK_PREEXISTING_BALANCE", "true"
).lower() in ("1","true","yes","y")

SLEEP_SEC     = float(os.environ.get("SLEEP_SEC", "60"))
MAX_RETRIES   = int(os.environ.get("MAX_RETRIES", "5"))
BACKOFF_BASE  = float(os.environ.get("BACKOFF_BASE", "1.5"))
REQ_INTERVAL  = float(os.environ.get("REQUEST_INTERVAL_SEC", "1.0"))

# Anti-double trade
COOLDOWN_SEC      = int(os.environ.get("COOLDOWN_SEC", "75"))
ORDER_ID_PREFIX   = os.environ.get("ORDER_ID_PREFIX", "xrpbot")
ONE_POSITION_ONLY = os.environ.get("ONE_POSITION_ONLY", "false").lower() in ("1","true","yes","y")
MAX_BUYS_PER_POSITION = int(os.environ.get("MAX_BUYS_PER_POSITION", "1"))

API_KEY    = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
OPERATOR_ID = os.environ.get("OPERATOR_ID", "").strip()

print("API_KEY    =", _mask(API_KEY))
print("API_SECRET =", _mask(API_SECRET))
print("SYMBOL     =", SYMBOL)

if not LIVE:
    raise SystemExit("Deze versie is LIVE-only. Zet LIVE=true in Secrets.")
if not API_KEY or not API_SECRET:
    raise SystemExit("GEEN API SLEUTELS: zet API_KEY en API_SECRET in Secrets.")

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
def log_trade(ts_iso: str, mode: str, side: str, price: float, amount: float, reason: str):
    new_file = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","mode","side","price","amount","reason"])
        w.writerow([ts_iso, mode, side, f"{price:.8f}", f"{amount:.8f}", reason])

# ---------- exchange ----------
ex = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
})
# bitvavo accepteert 'cost' voor market buy, prijs niet nodig
ex.options["createMarketBuyOrderRequiresPrice"] = False

markets = ex.load_markets()
if SYMBOL not in markets:
    raise SystemExit(f"Symbool {SYMBOL} niet gevonden op Bitvavo.")

BASE, QUOTE = SYMBOL.split("/")  # bv. XRP / EUR
_limits   = markets[SYMBOL].get("limits", {})
MIN_BASE  = float((_limits.get("amount") or {}).get("min") or 0.0)   # min in BASE
MIN_QUOTE = float((_limits.get("cost")   or {}).get("min") or 0.0)   # min in EUR
print(f"[INFO] Minima {SYMBOL}: min_cost=€{MIN_QUOTE}, min_amount={MIN_BASE} {BASE}")

def amount_to_precision(symbol: str, amount: float) -> float:
    return float(ex.amount_to_precision(symbol, amount))

# ---------- minimaal rate-limit respecteren voor lees-calls ----------
_last_call_ts = 0.0
def _respect_min_interval():
    global _last_call_ts
    now = time.time()
    wait = REQ_INTERVAL - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()

def read_retry(call, *args, **kwargs):
    delay = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _respect_min_interval()
            return call(*args, **kwargs)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            jitter = random.uniform(0, 0.5)
            sleep_s = delay + jitter
            print(f"[retry {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e} -> sleep {sleep_s:.2f}s")
            bot_logger.warning("retry %s/%s: %s", attempt, MAX_RETRIES, e)
            time.sleep(sleep_s)
            delay *= BACKOFF_BASE
        except Exception as e:
            print(f"[retry {attempt}/{MAX_RETRIES}] Unexpected error: {e}")
            bot_logger.exception("unexpected: %s", e)
            time.sleep(1.0)
    raise RuntimeError("Max retries reached for read call.")

# ---------- data & signaal (SMA cross) ----------
def fetch(limit=SLOW + 6) -> pd.DataFrame:
    ohlcv = read_retry(ex.fetch_ohlcv, SYMBOL, timeframe=TIMEFRAME, limit=limit)
    return pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])

def signal_from_df(df: pd.DataFrame) -> tuple[str, int, float]:
    """
    Retourneert (sig, closed_candle_ts, closed_price)
    We handelen op de **gesloten** candle (index -2).
    """
    if len(df) < max(FAST, SLOW) + 3:
        return "HOLD", int(df["t"].iloc[-1]), float(df["c"].iloc[-1])
    c = df["c"]
    f = c.rolling(FAST).mean()
    s = c.rolling(SLOW).mean()
    # Gesloten candle = -2
    sig = "HOLD"
    if (f.iloc[-2] > s.iloc[-2]) and (f.iloc[-3] <= s.iloc[-3]):
        sig = "BUY"
    elif (f.iloc[-2] < s.iloc[-2]) and (f.iloc[-3] >= s.iloc[-3]):
        sig = "SELL"
    return sig, int(df["t"].iloc[-2]), float(df["c"].iloc[-1])

# ---------- idempotente order helpers ----------
def make_client_id(side: str, candle_ms: int) -> str:
    # 1 id per kant per candle (idempotent bij retries/herstarts)
    return f"{ORDER_ID_PREFIX}-{int(candle_ms/1000)}-{side.lower()}"

def safe_create_market_order(symbol: str, side: str, amount: float|None, cost: float|None, client_id: str):
    """
    Geen agressieve retry; 1 zachte retry met **dezelfde** clientOrderId.
    Als duplicate -> fetch_orders en return de bestaande.
    """
    params = {"clientOrderId": client_id}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    if cost is not None:
        params["cost"] = float(round(cost, 2))

    try:
        return ex.create_order(symbol, "market", side, amount, None, params)
    except ccxt.NetworkError:
        time.sleep(0.8)
        return ex.create_order(symbol, "market", side, amount, None, params)
    except ccxt.ExchangeError as e:
        msg = str(e).lower()
        if "clientorderid" in msg and ("exists" in msg or "already" in msg or "duplicate" in msg):
            ords = read_retry(ex.fetch_orders, symbol, limit=10)
            for o in ords:
                if o.get("clientOrderId") == client_id:
                    return o
        raise

# ---------- startbalans lock ----------
start_bal = read_retry(ex.fetch_balance).get("free", {})
start_eur  = float(start_bal.get(QUOTE, 0) or 0)
start_base = float(start_bal.get(BASE, 0)  or 0)

# ---------- trade state ----------
entry_price: float | None = None
position_amt: float = 0.0
buys_in_position: int = 0
EPS = 1e-8

# trailing state
trail_active = False
trail_peak   = None
trail_floor  = None

def reset_trailing():
    global trail_active, trail_peak, trail_floor
    trail_active = False
    trail_peak = None
    trail_floor = None

# anti-double
last_candle_ts: int | None = None
last_fired_side: str | None = None
last_trade_ts: float = 0.0

# veiligheids-clamp
if EUR_PER_TRADE > MAX_EUR_PER_TRADE:
    print(f"[SAFEGUARD] Clamp €/trade {EUR_PER_TRADE} -> {MAX_EUR_PER_TRADE}")
    EUR_PER_TRADE = MAX_EUR_PER_TRADE

print(
    f"Bot gestart | {SYMBOL} | TF={TIMEFRAME} | LIVE={LIVE} | €/trade={EUR_PER_TRADE} "
    f"| SL={STOP_LOSS*100:.1f}% | TP={TAKE_PROFIT*100:.1f}% | CAP={MAX_EUR_PER_TRADE}"
)
bot_logger.info(
    "START | symbol=%s tf=%s live=%s eur_per_trade=%.2f sl=%.3f tp=%.3f cap=%.2f",
    SYMBOL, TIMEFRAME, LIVE, EUR_PER_TRADE, STOP_LOSS, TAKE_PROFIT, MAX_EUR_PER_TRADE
)

# ============================== MAIN LOOP ==============================
while True:
    try:
        df = fetch()
        sig, closed_ts, px = signal_from_df(df)
        print(f"Prijs: €{px:.4f} | Signaal: {sig}")

        bal = read_retry(ex.fetch_balance)
        free = bal.get("free", {})
        eur_free  = float(free.get(QUOTE, 0) or 0)
        base_free = float(free.get(BASE, 0)  or 0)

        if LOCK_PREEXISTING_BALANCE:
            eur_for_bot  = max(0.0, eur_free  - start_eur)
            base_for_bot = max(0.0, base_free - start_base)
        else:
            eur_for_bot  = eur_free
            base_for_bot = base_free

        print(f"[DEBUG] Vrij {QUOTE} (bot): {eur_for_bot:.2f} | Vrij {BASE} (bot): {base_for_bot:.6f}")

        # ---- per-candle enkelvoud & cooldown
        new_candle = (closed_ts != last_candle_ts)
        if new_candle:
            last_fired_side = None
            last_candle_ts = closed_ts

        def cooldown_blocking() -> bool:
            left = COOLDOWN_SEC - (time.time() - last_trade_ts)
            if left > 0:
                print(f"[COOLDOWN] Nog {left:.0f}s…")
                return True
            return False

        # ---- trailing / SL / TP (mag buiten cooldown om, maar blijft idempotent door clientOrderId)
        if entry_price and position_amt > EPS:
            change_pct = (px - entry_price) / entry_price

            # trailing activeren
            if not trail_active and change_pct >= TRAIL_TRIGGER:
                trail_active = True
                trail_peak   = px
                start_lock   = entry_price * (1 + TRAIL_START)
                trail_floor  = max(start_lock, trail_peak * (1 - TRAIL_GAP))
                print(f"[TRAIL] Activated. peak=€{trail_peak:.6f} floor=€{trail_floor:.6f}")

            if trail_active:
                if px > trail_peak:
                    trail_peak = px
                    trail_floor = trail_peak * (1 - TRAIL_GAP)
                if px <= trail_floor:
                    to_sell = min(position_amt, base_for_bot, MAX_EUR_PER_TRADE/px)
                    to_sell = amount_to_precision(SYMBOL, to_sell)
                    if to_sell >= MIN_BASE and (to_sell * px) >= max(MIN_QUOTE, 0.50):
                        cid = make_client_id("sell", closed_ts)
                        order = safe_create_market_order(SYMBOL, "sell", to_sell, None, cid)
                        print(f"[TRAIL] SELL {to_sell} {BASE} @ €{px:.6f}")
                        bot_logger.info("TRAIL SELL %s %s @ %.6f", to_sell, BASE, px)
                        log_trade(now_iso(), "LIVE", "SELL", px, to_sell, "TRAIL")
                        position_amt = max(0.0, position_amt - to_sell)
                        last_trade_ts = time.time()
                        last_fired_side = "SELL"
                        if position_amt <= EPS:
                            entry_price = None
                            buys_in_position = 0
                            reset_trailing()

            # vaste SL/TP
            if position_amt > EPS and (change_pct >= TAKE_PROFIT or change_pct <= -STOP_LOSS):
                reason = "TP" if change_pct >= TAKE_PROFIT else "SL"
                to_sell = min(position_amt, base_for_bot, MAX_EUR_PER_TRADE/px)
                to_sell = amount_to_precision(SYMBOL, to_sell)
                if to_sell >= MIN_BASE and (to_sell * px) >= max(MIN_QUOTE, 0.50):
                    cid = make_client_id("sell", closed_ts)
                    order = safe_create_market_order(SYMBOL, "sell", to_sell, None, cid)
                    print(f"[LIVE] {reason} -> SELL {to_sell} {BASE} @ €{px:.6f}")
                    bot_logger.info("%s SELL %s %s @ %.6f", reason, to_sell, BASE, px)
                    log_trade(now_iso(), "LIVE", "SELL", px, to_sell, reason)
                    position_amt = max(0.0, position_amt - to_sell)
                    last_trade_ts = time.time()
                    last_fired_side = "SELL"
                    if position_amt <= EPS:
                        entry_price = None
                        buys_in_position = 0
                        reset_trailing()

        # ---- SIGNAALEL: BUY / SELL (één per candle + cooldown + idempotent)
        if sig == "BUY":
            if ONE_POSITION_ONLY and position_amt > EPS:
                print("[SKIP] ONE_POSITION_ONLY actief en er is al positie.")
            elif buys_in_position >= MAX_BUYS_PER_POSITION:
                print("[SKIP] MAX_BUYS_PER_POSITION bereikt.")
            elif last_fired_side == "BUY":
                print("[SKIP] BUY al gedaan deze candle.")
            elif cooldown_blocking():
                pass
            else:
                eur_room = eur_for_bot
                if eur_room <= 0:
                    print(f"[LIVE] Geen {QUOTE} beschikbaar (boven startbalans).")
                else:
                    # minima
                    min_required_eur = max(MIN_QUOTE, MIN_BASE * px, 0.50)
                    target = min(EUR_PER_TRADE, MAX_EUR_PER_TRADE, eur_room)
                    if target + 1e-9 < min_required_eur:
                        new_target = min(max(min_required_eur, EUR_PER_TRADE), MAX_EUR_PER_TRADE, eur_room)
                        if new_target + 1e-9 < min_required_eur:
                            print(f"[LIVE] BUY te klein (nodig ≥ €{min_required_eur:.2f}).")
                            time.sleep(SLEEP_SEC); continue
                        target = new_target
                    spend_eur = round(min(target, eur_room, MAX_EUR_PER_TRADE), 2)

                    est_base = spend_eur / px
                    if (MIN_BASE > 0 and est_base + 1e-9 < MIN_BASE):
                        print("[LIVE] BUY overslaan: nog onder min amount.")
                        time.sleep(SLEEP_SEC); continue

                    cid = make_client_id("buy", closed_ts)
                    order = safe_create_market_order(SYMBOL, "buy", None, spend_eur, cid)

                    # filled-hoeveelheid
                    filled = None
                    try:
                        filled = float(order.get("info", {}).get("filledAmount"))
                    except Exception:
                        pass
                    if not filled:
                        filled = spend_eur / px

                    position_amt += filled
                    buys_in_position += 1
                    if entry_price is None:
                        entry_price = px
                        reset_trailing()

                    print(f"[LIVE] BUY ~€{spend_eur:.2f} {BASE} (+{filled:.6f}) @ €{px:.6f}")
                    bot_logger.info("BUY €%.2f -> +%s %s @ %.6f", spend_eur, f"{filled:.6f}", BASE, px)
                    log_trade(now_iso(), "LIVE", "BUY", px, spend_eur, "SIGNAL")
                    last_trade_ts = time.time()
                    last_fired_side = "BUY"

        elif sig == "SELL":
            if last_fired_side == "SELL":
                print("[SKIP] SELL al gedaan deze candle.")
            elif position_amt <= EPS or base_for_bot <= 0:
                print("[LIVE] Niets van deze sessie om te verkopen.")
            elif cooldown_blocking():
                pass
            else:
                to_sell = min(position_amt, base_for_bot, MAX_EUR_PER_TRADE/px)
                to_sell = amount_to_precision(SYMBOL, to_sell)
                if to_sell >= MIN_BASE and (to_sell * px) >= max(MIN_QUOTE, 0.50):
                    cid = make_client_id("sell", closed_ts)
                    order = safe_create_market_order(SYMBOL, "sell", to_sell, None, cid)
                    position_amt = max(0.0, position_amt - to_sell)
                    msg = f"SIGNAL SELL {to_sell} {BASE} @ €{px:.6f}"
                    print("[LIVE]", msg); bot_logger.info(msg)
                    log_trade(now_iso(), "LIVE", "SELL", px, to_sell, "SIGNAL")
                    last_trade_ts = time.time()
                    last_fired_side = "SELL"
                    if position_amt <= EPS:
                        entry_price = None
                        buys_in_position = 0
                        reset_trailing()
                else:
                    print(f"[LIVE] SELL te klein (min {MIN_BASE:.6f} {BASE} en €{MIN_QUOTE}).")

        time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Gestopt."); bot_logger.info("STOP by user")
        break
    except Exception as e:
        print("[LIVE] ❌ Runtime error:", repr(e))
        bot_logger.exception("Runtime error: %s", e)
        time.sleep(5)
