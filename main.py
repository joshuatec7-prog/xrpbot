# ==============================
#  xrp_minibot – LIVE only (Bitvavo + trailing stop + lock balances)
# ==============================

import os, time, csv, pathlib, logging, random
from logging.handlers import RotatingFileHandler
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import ccxt

from keep_alive import keep_alive
keep_alive()  # webserver voor uptime pings (Replit)

# ---------- helpers ----------
def now_iso() -> str:
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%d-%m-%Y %H:%M:%S")

def _mask(s: str | None) -> str:
    if not s: return "MISSING"
    return s[:4] + "..." + s[-4:]

# ========== Instellingen uit Secrets ==========
SYMBOL    = os.environ.get("SYMBOL", "XRP/EUR")
TIMEFRAME = os.environ.get("TIMEFRAME", "1m")
FAST      = int(os.environ.get("FAST", "3"))
SLOW      = int(os.environ.get("SLOW", "6"))
LIVE      = os.environ.get("LIVE", "true").lower() in ("1","true","yes","y")

EUR_PER_TRADE     = float(os.environ.get("EUR_PER_TRADE", "6"))
MAX_EUR_PER_TRADE = float(os.environ.get("MAX_EUR_PER_TRADE", "10"))

# Risk settings
STOP_LOSS   = float(os.environ.get("STOP_LOSS", "0.01"))    # 1% vaste SL
TAKE_PROFIT = float(os.environ.get("TAKE_PROFIT", "0.03"))  # 3% vaste TP

# Trailing stop params
TRAIL_TRIGGER = float(os.environ.get("TRAIL_TRIGGER", "0.02"))  # activeer bij +2%
TRAIL_START   = float(os.environ.get("TRAIL_START", "0.015"))   # eerste lock bij +1.5%
TRAIL_GAP     = float(os.environ.get("TRAIL_GAP", "0.01"))      # trailing 1% onder piek

LOCK_PREEXISTING_BALANCE = os.environ.get(
    "LOCK_PREEXISTING_BALANCE", "true"
).lower() in ("1","true","yes","y")

SLEEP_SEC     = float(os.environ.get("SLEEP_SEC", "60"))
MAX_RETRIES   = int(os.environ.get("MAX_RETRIES", "5"))
BACKOFF_BASE  = float(os.environ.get("BACKOFF_BASE", "1.5"))
REQ_INTERVAL  = float(os.environ.get("REQUEST_INTERVAL_SEC", "1.0"))

API_KEY     = os.environ.get("API_KEY", "")
API_SECRET  = os.environ.get("API_SECRET", "")
OPERATOR_ID = os.environ.get("OPERATOR_ID", "").strip()

print("API_KEY    =", _mask(API_KEY))
print("API_SECRET =", _mask(API_SECRET))
print("SYMBOL     =", SYMBOL)

if not LIVE:
    raise SystemExit("Deze versie is LIVE-only. Zet LIVE=true in Secrets.")
if not API_KEY or not API_SECRET:
    raise SystemExit("GEEN API SLEUTELS: zet API_KEY en API_SECRET in Secrets.")

# ========== Logging ==========
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

# ========== Exchange ==========
ex = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
})
ex.options["createMarketBuyOrderRequiresPrice"] = False  # BUY via 'cost'

# Markets + minima
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

# ========== Retry + rate-limit ==========
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
    raise RuntimeError("Max retries reached for exchange call.")

# ========== Data & Signaal ==========
def fetch(limit=SLOW + 3) -> pd.DataFrame:
    ohlcv = with_retry(ex.fetch_ohlcv, SYMBOL, timeframe=TIMEFRAME, limit=limit)
    return pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])

def signal(df: pd.DataFrame) -> str:
    if len(df) < max(FAST, SLOW) + 2:
        return "HOLD"
    f = df["c"].rolling(FAST).mean()
    s = df["c"].rolling(SLOW).mean()
    if f.iloc[-2] > s.iloc[-2] and f.iloc[-3] <= s.iloc[-3]:
        return "BUY"
    if f.iloc[-2] < s.iloc[-2] and f.iloc[-3] >= s.iloc[-3]:
        return "SELL"
    return "HOLD"

# ========== Veiligheids-clamp ==========
if EUR_PER_TRADE > MAX_EUR_PER_TRADE:
    print(f"[SAFEGUARD] EUR_PER_TRADE ({EUR_PER_TRADE}) > MAX_EUR_PER_TRADE ({MAX_EUR_PER_TRADE}). Clamp.")
    EUR_PER_TRADE = MAX_EUR_PER_TRADE

print(
    f"Bot gestart | {SYMBOL} | TF={TIMEFRAME} | LIVE={LIVE} | €/trade={EUR_PER_TRADE} "
    f"| SL={STOP_LOSS*100:.1f}% | TP={TAKE_PROFIT*100:.1f}% | CAP={MAX_EUR_PER_TRADE}"
)
bot_logger.info(
    "START | symbol=%s tf=%s live=%s eur_per_trade=%.2f sl=%.3f tp=%.3f cap=%.2f",
    SYMBOL, TIMEFRAME, LIVE, EUR_PER_TRADE, STOP_LOSS, TAKE_PROFIT, MAX_EUR_PER_TRADE
)

# ========== Startbalans (lock) ==========
start_bal = with_retry(ex.fetch_balance).get("free", {})
start_eur  = float(start_bal.get(QUOTE, 0) or 0)
start_base = float(start_bal.get(BASE, 0) or 0)

# ========== Trade state ==========
entry_price: float | None = None
position_amt: float = 0.0   # hoeveelheid die deze bot heeft opgebouwd (boven start)
EPS = 1e-8

# Trailing stop state
trail_active = False
trail_peak   = None   # hoogste prijs sinds trail start
trail_floor  = None   # verkoopgrens = peak * (1 - TRAIL_GAP)

def reset_trailing():
    global trail_active, trail_peak, trail_floor
    trail_active = False
    trail_peak = None
    trail_floor = None

# ========== Hoofdloop ==========
while True:
    try:
        df = fetch()
        px = float(df["c"].iloc[-1])
        sig = signal(df)
        print(f"Prijs: €{px:.4f} | Signaal: {sig}")

        bal = with_retry(ex.fetch_balance)
        free = bal.get("free", {})
        eur_free  = float(free.get(QUOTE, 0) or 0)
        base_free = float(free.get(BASE, 0) or 0)

        # Beschikbaar voor de bot (respecteer lock)
        if LOCK_PREEXISTING_BALANCE:
            eur_for_bot  = max(0.0, eur_free  - start_eur)
            base_for_bot = max(0.0, base_free - start_base)
        else:
            eur_for_bot  = eur_free
            base_for_bot = base_free

        print(f"[DEBUG] Vrij {QUOTE} (bot): {eur_for_bot:.2f} | Vrij {BASE} (bot): {base_for_bot:.6f}")

        # -------- SL/TP + Trailing --------
        if entry_price and position_amt > EPS:
            change_pct = (px - entry_price) / entry_price

            # 1) Trailing: activeren bij +TRAIL_TRIGGER
            if not trail_active and change_pct >= TRAIL_TRIGGER:
                trail_active = True
                trail_peak  = px
                # init floor op max(entry*(1+TRAIL_START), peak*(1-TRAIL_GAP))
                start_lock  = entry_price * (1 + TRAIL_START)
                trail_floor = max(start_lock, trail_peak * (1 - TRAIL_GAP))
                print(f"[TRAIL] Activated. peak=€{trail_peak:.6f} floor=€{trail_floor:.6f}")

            # 2) Trailing bijhouden
            if trail_active:
                if px > trail_peak:
                    trail_peak = px
                    trail_floor = trail_peak * (1 - TRAIL_GAP)
                if px <= trail_floor:
                    # verkoop via trailing
                    to_sell = min(position_amt, base_for_bot)
                    to_sell = amount_to_precision(SYMBOL, to_sell)
                    if to_sell >= MIN_BASE and (to_sell * px) >= max(MIN_QUOTE, 0.50):
                        params = {}
                        if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
                        with_retry(ex.create_order, SYMBOL, "market", "sell", to_sell, None, params)
                        print(f"[TRAIL] SELL {to_sell} {BASE} @ €{px:.6f} (peak=€{trail_peak:.6f})")
                        bot_logger.info("TRAIL SELL %s %s @ %.6f", to_sell, BASE, px)
                        log_trade(now_iso(), "LIVE", "SELL", px, to_sell, "TRAILING_STOP")
                        position_amt = max(0.0, position_amt - to_sell)
                        if position_amt <= EPS:
                            entry_price = None
                            reset_trailing()
                    else:
                        print("[TRAIL] Signaal, maar te klein voor beurs-minima. Overslaan.")

            # 3) Backstops: vaste TP/SL
            if position_amt > EPS and px and (change_pct >= TAKE_PROFIT or change_pct <= -STOP_LOSS):
                reason = "TAKE_PROFIT" if change_pct >= TAKE_PROFIT else "STOP_LOSS"
                to_sell = min(position_amt, base_for_bot)
                to_sell = amount_to_precision(SYMBOL, to_sell)
                if to_sell >= MIN_BASE and (to_sell * px) >= max(MIN_QUOTE, 0.50):
                    params = {}
                    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
                    with_retry(ex.create_order, SYMBOL, "market", "sell", to_sell, None, params)
                    msg = f"{reason} {change_pct*100:+.2f}% -> SELL {to_sell} {BASE} @ €{px:.6f}"
                    print("[LIVE]", msg); bot_logger.info(msg)
                    log_trade(now_iso(), "LIVE", "SELL", px, to_sell, reason)
                    position_amt = max(0.0, position_amt - to_sell)
                    if position_amt <= EPS:
                        entry_price = None
                        reset_trailing()
                else:
                    print("[LIVE] SL/TP signaal maar te klein voor minima. Overslaan.")

        # --------------------------- BUY ---------------------------
        if sig == "BUY":
            # koop alleen met bot-EUR (lock)
            eur_room = eur_for_bot
            if eur_room <= 0:
                print(f"[LIVE] Geen {QUOTE} beschikbaar (boven startbalans).")
            else:
                target = min(EUR_PER_TRADE, MAX_EUR_PER_TRADE, eur_room)
                required_eur_for_amount = (MIN_BASE * px) if MIN_BASE > 0 else 0.0
                min_required_eur = max(MIN_QUOTE, required_eur_for_amount, 0.50)

                # indien nodig iets ophogen binnen je cap en room
                if target + 1e-9 < min_required_eur:
                    new_target = min(max(min_required_eur, EUR_PER_TRADE), MAX_EUR_PER_TRADE, eur_room)
                    if new_target + 1e-9 < min_required_eur:
                        print(f"[LIVE] BUY overslaan: bedrag te klein (nodig ≥ max({MIN_BASE:.6f} {BASE}, €{min_required_eur:.2f})).")
                        time.sleep(SLEEP_SEC); continue
                    target = new_target

                spend_eur = round(min(target, eur_room, MAX_EUR_PER_TRADE), 2)
                est_base  = spend_eur / px
                if (MIN_BASE > 0 and est_base + 1e-9 < MIN_BASE) or spend_eur + 1e-9 < max(MIN_QUOTE, 0.50):
                    print("[LIVE] BUY overslaan (na controle): nog steeds onder minima.")
                    time.sleep(SLEEP_SEC); continue

                params = {"cost": spend_eur}
                if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
                order = with_retry(ex.create_order, SYMBOL, "market", "buy", None, None, params)

                filled = None
                try:
                    filled = float(order.get("info", {}).get("filledAmount"))
                except Exception:
                    pass
                if not filled:
                    filled = spend_eur / px

                position_amt += filled
                entry_price = px
                reset_trailing()  # nieuwe trade -> trailing opnieuw

                msg = f"BUY ~€{spend_eur:.2f} {BASE} (+{filled:.6f}) @ €{px:.6f}"
                print("[LIVE]", msg); bot_logger.info(msg)
                log_trade(now_iso(), "LIVE", "BUY", px, spend_eur, "SIGNAL")

        # --------------------------- SELL --------------------------
        elif sig == "SELL":
            if position_amt > EPS and base_for_bot > 0:
                max_eur = MAX_EUR_PER_TRADE / px
                to_sell = min(position_amt, base_for_bot, max_eur)
                to_sell = amount_to_precision(SYMBOL, to_sell)
                if to_sell >= MIN_BASE and (to_sell * px) >= max(MIN_QUOTE, 0.50):
                    params = {}
                    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
                    with_retry(ex.create_order, SYMBOL, "market", "sell", to_sell, None, params)
                    position_amt = max(0.0, position_amt - to_sell)
                    if position_amt <= EPS:
                        entry_price = None
                        reset_trailing()
                    msg = f"SIGNAL SELL {to_sell} {BASE} @ €{px:.6f}"
                    print("[LIVE]", msg); bot_logger.info(msg)
                    log_trade(now_iso(), "LIVE", "SELL", px, to_sell, "SIGNAL")
                else:
                    print(f"[LIVE] SELL signaal maar hoeveelheid te klein (min {MIN_BASE:.6f} {BASE} en €{MIN_QUOTE}). Overslaan.")
            else:
                print("[LIVE] Niets van deze sessie om te verkopen.")

        time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Gestopt."); bot_logger.info("STOP by user")
        break
    except Exception as e:
        print("[LIVE] ❌ Runtime error:", repr(e))
        bot_logger.exception("Runtime error: %s", e)
        time.sleep(5)
