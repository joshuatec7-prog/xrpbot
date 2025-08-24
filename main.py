# ==============================
#  Grid/Signal Bot – Render versie
# ==============================

import os, time, csv, pathlib, logging, random
from logging.handlers import RotatingFileHandler
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import ccxt

# --------- Helpers ----------
def now_iso() -> str:
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M:%S")

def _mask(s: str | None) -> str:
    if not s: return "MISSING"
    return s[:4] + "..." + s[-4:]

# --------- Config uit ENV ---------
SYMBOL    = os.environ.get("SYMBOL", "ETH/EUR")
TIMEFRAME = os.environ.get("TIMEFRAME", "1m")
FAST      = int(os.environ.get("FAST", "3"))
SLOW      = int(os.environ.get("SLOW", "6"))

EUR_PER_TRADE     = float(os.environ.get("EUR_PER_TRADE", "15"))
MAX_EUR_PER_TRADE = float(os.environ.get("MAX_EUR_PER_TRADE", "20"))

STOP_LOSS   = float(os.environ.get("STOP_LOSS", "0.01"))    # 1%
TAKE_PROFIT = float(os.environ.get("TAKE_PROFIT", "0.03"))  # 3%

TRAIL_TRIGGER = float(os.environ.get("TRAIL_TRIGGER", "0.02"))
TRAIL_START   = float(os.environ.get("TRAIL_START", "0.015"))
TRAIL_GAP     = float(os.environ.get("TRAIL_GAP", "0.01"))

ONE_POSITION_ONLY = os.environ.get("ONE_POSITION_ONLY", "true").lower() in ("1","true","yes")
MAX_BUYS_PER_POSITION = int(os.environ.get("MAX_BUYS_PER_POSITION", "1"))
TRADE_COOLDOWN_SEC = int(os.environ.get("TRADE_COOLDOWN_SEC", "180"))

SLEEP_SEC   = float(os.environ.get("SLEEP_SEC", "60"))
REQ_INTERVAL = float(os.environ.get("REQUEST_INTERVAL_SEC", "1.0"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "1.5"))

API_KEY    = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
ORDER_ID_PREFIX = os.environ.get("ORDER_ID_PREFIX", "gridbot")

if not API_KEY or not API_SECRET:
    raise SystemExit("❌ API-sleutels ontbreken, zet API_KEY en API_SECRET in Secrets.")

print("API_KEY   =", _mask(API_KEY))
print("SYMBOL    =", SYMBOL)

# --------- Logging ----------
LOG_DIR = pathlib.Path(".")
LOG_DIR.mkdir(exist_ok=True)
bot_logger = logging.getLogger("bot")
bot_logger.setLevel(logging.INFO)
if not bot_logger.handlers:
    h = RotatingFileHandler(LOG_DIR / "bot.log", maxBytes=512_000, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    bot_logger.addHandler(h)

TRADES_CSV = LOG_DIR / "trades.csv"
def log_trade(ts_iso, side, price, amount, reason):
    new_file = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","side","price","amount","reason"])
        w.writerow([ts_iso, side, f"{price:.2f}", f"{amount:.6f}", reason])

# --------- Exchange ----------
ex = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
})
ex.options["createMarketBuyOrderRequiresPrice"] = False

markets = ex.load_markets()
if SYMBOL not in markets:
    raise SystemExit(f"❌ Symbool {SYMBOL} niet gevonden op Bitvavo.")

BASE, QUOTE = SYMBOL.split("/")
_limits = markets[SYMBOL].get("limits", {})
MIN_BASE  = float((_limits.get("amount") or {}).get("min") or 0.0)
MIN_QUOTE = float((_limits.get("cost")   or {}).get("min") or 0.0)

# --------- Trading state ----------
entry_price = None
position_amt = 0.0
last_trade_time = 0

# --------- Functies ----------
def fetch(limit=SLOW+3) -> pd.DataFrame:
    ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
    return pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])

def signal(df: pd.DataFrame) -> str:
    f = df["c"].rolling(FAST).mean()
    s = df["c"].rolling(SLOW).mean()
    if f.iloc[-2] > s.iloc[-2] and f.iloc[-3] <= s.iloc[-3]:
        return "BUY"
    if f.iloc[-2] < s.iloc[-2] and f.iloc[-3] >= s.iloc[-3]:
        return "SELL"
    return "HOLD"

def can_trade():
    global last_trade_time
    return (time.time() - last_trade_time) >= TRADE_COOLDOWN_SEC

# --------- Hoofdloop ----------
print(f"🚀 Bot gestart op {SYMBOL}, €/trade={EUR_PER_TRADE}, cooldown={TRADE_COOLDOWN_SEC}s")

while True:
    try:
        df = fetch()
        px = float(df["c"].iloc[-1])
        sig = signal(df)
        print(f"Prijs €{px:.2f} | Signaal {sig}")

        bal = ex.fetch_balance()
        free = bal.get("free", {})
        eur_free = float(free.get(QUOTE, 0) or 0)
        base_free = float(free.get(BASE, 0) or 0)

        if sig == "BUY" and can_trade():
            if ONE_POSITION_ONLY and position_amt > 0:
                print("⏸️ Al positie open → geen extra BUY")
            else:
                spend = min(EUR_PER_TRADE, eur_free, MAX_EUR_PER_TRADE)
                if spend >= MIN_QUOTE:
                    order = ex.create_order(SYMBOL, "market", "buy", None, {"cost": spend})
                    amount = spend / px
                    position_amt += amount
                    entry_price = px
                    last_trade_time = time.time()
                    print(f"✅ BUY {amount:.6f} {BASE} voor ~€{spend:.2f}")
                    log_trade(now_iso(), "BUY", px, amount, "SIGNAL")

        elif sig == "SELL" and position_amt > 0 and can_trade():
            to_sell = min(position_amt, base_free)
            if to_sell * px >= MIN_QUOTE:
                ex.create_order(SYMBOL, "market", "sell", to_sell)
                position_amt = 0
                entry_price = None
                last_trade_time = time.time()
                print(f"✅ SELL {to_sell:.6f} {BASE} @ €{px:.2f}")
                log_trade(now_iso(), "SELL", px, to_sell, "SIGNAL")

        time.sleep(SLEEP_SEC)

    except Exception as e:
        print("⚠️ Error:", e)
        bot_logger.exception("Runtime error: %s", e)
        time.sleep(5)
