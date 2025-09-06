# main.py
# --- Spot grid/signal bot (Bitvavo) met strict "one-at-a-time" + TP/SL/Trailing ---
# - Kopen alleen als er GEEN open positie is (STRICT_ONE_AT_A_TIME)
# - Volledige exit (100%) bij TAKE_PROFIT of STOP_LOSS (en optioneel TRAILING_STOP)
# - Minimale order- en cost-checks voor Bitvavo
# - Netwerk-retries en interval respect
# - CSV-log van trades (trades.csv) + rolling bot.log

import os
import time
import csv
import random
import logging
import pathlib
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional

from zoneinfo import ZoneInfo
import ccxt
import pandas as pd

# ========== Helpers ==========
def now_iso() -> str:
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M:%S")

def mask(s: Optional[str]) -> str:
    if not s:
        return "MISSING"
    if len(s) <= 8:
        return s
    return s[:4] + "..." + s[-4:]


# ========== ENV ==========
API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

SYMBOL    = os.getenv("SYMBOL", "ETH/EUR")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")

# Signaal (MA cross)
FAST = int(os.getenv("FAST", "3"))
SLOW = int(os.getenv("SLOW", "6"))

# Live mode
LIVE = os.getenv("LIVE", "true").lower() in ("1", "true", "yes")

# Trade sizing
EUR_PER_TRADE     = float(os.getenv("EUR_PER_TRADE", "15"))
MAX_EUR_PER_TRADE = float(os.getenv("MAX_EUR_PER_TRADE", "20"))

# Fees & winst/risico
FEE_PCT        = float(os.getenv("FEE_PCT", "0.0015"))         # 0,15% per order
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.006"))   # 0,6% netto target
TAKE_PROFIT    = float(os.getenv("TAKE_PROFIT", "0.010"))      # 1,0% hard TP
STOP_LOSS      = float(os.getenv("STOP_LOSS", "0.03"))         # 3% SL
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.0"))  # 0.0 = uit

# Cooldowns & tempo
TRADE_COOLDOWN_SEC   = int(os.getenv("TRADE_COOLDOWN_SEC", "300"))
SLEEP_SEC            = float(os.getenv("SLEEP_SEC", "60"))
REQUEST_INTERVAL_SEC = float(os.getenv("REQUEST_INTERVAL_SEC", "1.0"))
MAX_RETRIES          = int(os.getenv("MAX_RETRIES", "5"))
BACKOFF_BASE         = float(os.getenv("BACKOFF_BASE", "1.5"))

# Gedrag
ONE_POSITION_ONLY = os.getenv("ONE_POSITION_ONLY", "true").lower() in ("1","true","yes")
STRICT_ONE_AT_A_TIME = os.getenv("STRICT_ONE_AT_A_TIME", "true").lower() in ("1","true","yes")
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE", "false").lower() in ("1","true","yes")

# Overig (optioneel)
ORDER_ID_PREFIX = os.getenv("ORDER_ID_PREFIX", "gridbot")
OPERATOR_ID     = os.getenv("OPERATOR_ID", "").strip()

LOG_TRADES      = os.getenv("LOG_TRADES", "true").lower() in ("1","true","yes")
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC", "600"))

# ========== Logging ==========
LOG_DIR = pathlib.Path(".")
LOG_DIR.mkdir(exist_ok=True)
botlog = logging.getLogger("bot")
botlog.setLevel(logging.INFO)
if not botlog.handlers:
    h = RotatingFileHandler(LOG_DIR / "bot.log", maxBytes=512_000, backupCount=3, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    botlog.addHandler(h)

TRADES_CSV = LOG_DIR / "trades.csv"
def log_trade(ts, side, price, amount, reason, pnl_eur=""):
    if not LOG_TRADES:
        return
    new = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp","symbol","side","price","amount","reason","pnl_eur"])
        w.writerow([ts, SYMBOL, side, f"{price:.8f}", f"{amount:.8f}", reason, pnl_eur])

# ========== Checks ==========
if not LIVE:
    raise SystemExit("Deze bot draait alleen in LIVE modus. Zet LIVE=true in je Env Vars.")
if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY en/of API_SECRET ontbreekt.")

print(f"API_KEY={mask(API_KEY)} | API_SECRET={mask(API_SECRET)}")
print(
    f"Start | {SYMBOL} | tf={TIMEFRAME} | €/trade={EUR_PER_TRADE} | fee={FEE_PCT*100:.2f}% "
    f"| minProfit={MIN_PROFIT_PCT*100:.2f}% | TP={TAKE_PROFIT*100:.2f}% | SL={STOP_LOSS*100:.2f}% "
    f"| trailing={TRAILING_STOP_PCT*100:.2f}%"
)

# ========== Exchange ==========
ex = ccxt.bitvavo({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
ex.options["createMarketBuyOrderRequiresPrice"] = False

markets = ex.load_markets()
if SYMBOL not in markets:
    raise SystemExit(f"Symbool {SYMBOL} niet gevonden op Bitvavo.")

BASE, QUOTE = SYMBOL.split("/")
_limits = markets[SYMBOL].get("limits", {})
MIN_BASE  = float((_limits.get("amount") or {}).get("min") or 0.0)  # min in BASE
MIN_QUOTE = float((_limits.get("cost") or {}).get("min") or 0.0)    # min in EUR

def amount_to_precision(sym, amt: float) -> float:
    return float(ex.amount_to_precision(sym, amt))

_last_call = 0.0
def respect_interval():
    """Respecteer REQUEST_INTERVAL_SEC tussen beursaanslagen."""
    global _last_call
    now = time.time()
    wait = REQUEST_INTERVAL_SEC - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()

def with_retry(fn, *a, **kw):
    delay = 1.0
    for i in range(1, MAX_RETRIES + 1):
        try:
            respect_interval()
            return fn(*a, **kw)
        except (ccxt.RateLimitExceeded, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            botlog.warning("retry %s/%s: %s", i, MAX_RETRIES, e)
            time.sleep(delay + random.random() * 0.5)
            delay *= BACKOFF_BASE
        except Exception as e:
            botlog.exception("unexpected: %s", e)
            time.sleep(1.0)
    raise RuntimeError("Max retries reached.")

# ========== Data & signal ==========
def fetch_df(limit=SLOW + 3) -> pd.DataFrame:
    ohlcv = with_retry(ex.fetch_ohlcv, SYMBOL, timeframe=TIMEFRAME, limit=limit)
    return pd.DataFrame(ohlcv, columns=["t","o","h","l","c","v"])

def signal(df: pd.DataFrame) -> str:
    """MA cross signaal op slot-1 (stabieler dan slot-0)."""
    if len(df) < max(FAST, SLOW) + 2:
        return "HOLD"
    f = df["c"].rolling(FAST).mean()
    s = df["c"].rolling(SLOW).mean()
    if f.iloc[-2] > s.iloc[-2] and f.iloc[-3] <= s.iloc[-3]:
        return "BUY"
    if f.iloc[-2] < s.iloc[-2] and f.iloc[-3] >= s.iloc[-3]:
        return "SELL"
    return "HOLD"

# ========== Balans & state ==========
entry_price: Optional[float] = None
position_amt: float = 0.0     # in BASE
last_trade_ts = 0.0           # cooldown
trail_high: Optional[float] = None

# startbalans (voor LOCK_PREEXISTING_BALANCE)
_start_bal = with_retry(ex.fetch_balance).get("free", {})
start_eur  = float(_start_bal.get(QUOTE, 0) or 0)
start_base = float(_start_bal.get(BASE, 0)  or 0)

def eur_for_bot() -> float:
    bal = with_retry(ex.fetch_balance).get("free", {})
    eur = float(bal.get(QUOTE, 0) or 0)
    if LOCK_PREEXISTING_BALANCE:
        return max(0.0, eur - start_eur)
    return eur

def base_for_bot() -> float:
    bal = with_retry(ex.fetch_balance).get("free", {})
    base = float(bal.get(BASE, 0) or 0)
    if LOCK_PREEXISTING_BALANCE:
        return max(0.0, base - start_base)
    return base

# ========== Netto PnL helper ==========
def net_change_pct(cur_px: float, entry: float, fee_pct: float) -> float:
    """Benadering van netto % na round-trip fees (2x)."""
    if not entry or entry <= 0:
        return 0.0
    gross = (cur_px - entry) / entry
    return gross - 2 * fee_pct

# ========== Order helpers ==========
def market_buy_eur(spend_eur: float, px: float) -> float:
    spend_eur = float(f"{spend_eur:.2f}")  # Bitvavo cost = 2 decimalen
    min_needed = max(MIN_QUOTE, 0.50)
    if spend_eur < min_needed:
        print(f"[BUY] overslaan: bedrag te klein (min €{min_needed:.2f}).")
        return 0.0
    params = {"cost": spend_eur}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    order = with_retry(ex.create_order, SYMBOL, "market", "buy", None, None, params)
    filled = None
    try:
        filled = float(order.get("info", {}).get("filledAmount"))
    except Exception:
        pass
    if not filled:
        filled = spend_eur / px
    return filled

def market_sell_amount(qty: float, px: float) -> float:
    qty = amount_to_precision(SYMBOL, qty)
    min_needed = max(MIN_QUOTE, 0.50)
    if qty < MIN_BASE or (qty * px) < min_needed:
        print(f"[SELL] overslaan: hoeveelheid te klein (min {MIN_BASE} {BASE}, €{min_needed:.2f}).")
        return 0.0
    params = {}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    with_retry(ex.create_order, SYMBOL, "market", "sell", qty, None, params)
    return qty

# Volledige exit helper
def sell_all_now(cur_px: float, reason: str) -> None:
    global position_amt, entry_price, last_trade_ts, trail_high
    qty = amount_to_precision(SYMBOL, position_amt)
    if qty <= 0:
        return
    sold = market_sell_amount(qty, cur_px)
    if sold > 0:
        pnl_eur = sold * (cur_px - (entry_price or cur_px)) - (sold*cur_px + sold*(entry_price or cur_px)) * FEE_PCT
        print(f"[LIVE] {reason} SELL {sold:.6f} {BASE} @ €{cur_px:.4f} | pnl≈€{pnl_eur:.2f}")
        botlog.info("%s SELL | px=%.6f", reason, cur_px)
        log_trade(now_iso(), "SELL", cur_px, sold, reason, f"{pnl_eur:.2f}")
        position_amt -= sold
        if position_amt <= 1e-8:
            position_amt = 0.0
            entry_price = None
            trail_high = None
        last_trade_ts = time.time()

# ========== Hoofdloop ==========
last_summary = 0.0

while True:
    try:
        df = fetch_df()
        px = float(df["c"].iloc[-1])
        sig = signal(df)

        # Periodieke samenvatting
        now = time.time()
        if now - last_summary >= LOG_SUMMARY_SEC:
            print(f"[SUMMARY] prijs=€{px:.3f} | pos={position_amt:.6f} {BASE} | entry={entry_price or 0:.4f} | vrij €={eur_for_bot():.2f}")
            last_summary = now

        # === ALTJD eerst exits (TP/SL/Trailing), ook tijdens cooldown ===
        if position_amt > 0 and entry_price:
            # trailing high bijhouden
            if TRAILING_STOP_PCT > 0.0:
                trail_high = px if trail_high is None else max(trail_high, px)

            net = net_change_pct(px, entry_price, FEE_PCT)

            # Hard take profit
            if net >= TAKE_PROFIT:
                sell_all_now(px, "TAKE_PROFIT")
                time.sleep(SLEEP_SEC)
                continue

            # Hard stop loss
            if net <= -STOP_LOSS:
                sell_all_now(px, "STOP_LOSS")
                time.sleep(SLEEP_SEC)
                continue

            # Trailing stop (optioneel)
            if TRAILING_STOP_PCT > 0.0 and trail_high:
                if px <= trail_high * (1.0 - TRAILING_STOP_PCT):
                    sell_all_now(px, "TRAILING_STOP")
                    time.sleep(SLEEP_SEC)
                    continue

        # === BUY-logica: alleen als GEEN positie ===
        if sig == "BUY":
            if STRICT_ONE_AT_A_TIME and position_amt > 1e-8:
                print("[LIVE] BUY geblokkeerd: positie open (STRICT_ONE_AT_A_TIME).")
                time.sleep(SLEEP_SEC)
                continue

            if ONE_POSITION_ONLY and position_amt > 0:
                print("[LIVE] BUY geblokkeerd: ONE_POSITION_ONLY.")
                time.sleep(SLEEP_SEC)
                continue

            if time.time() - last_trade_ts < TRADE_COOLDOWN_SEC:
                time.sleep(SLEEP_SEC)
                continue

            room = eur_for_bot()
            if room <= 0.5:
                print("[LIVE] Geen vrij EUR.")
                time.sleep(SLEEP_SEC)
                continue

            target = min(EUR_PER_TRADE, MAX_EUR_PER_TRADE, room)
            min_needed = max(MIN_QUOTE, 0.50)
            if target < min_needed:
                target = min(min_needed, room, MAX_EUR_PER_TRADE)
                if target < min_needed:
                    print(f"[LIVE] BUY overslaan: min kost €{min_needed:.2f}.")
                    time.sleep(SLEEP_SEC)
                    continue

            filled = market_buy_eur(target, px)
            if filled > 0:
                # nieuwe entry & trailing resetten
                entry_price = px
                trail_high = px if TRAILING_STOP_PCT > 0.0 else None
                position_amt += filled
                last_trade_ts = time.time()
                print(f"[LIVE] BUY ~€{target:.2f} -> {filled:.6f} {BASE} @ €{px:.4f} | pos={position_amt:.6f} | entry=€{entry_price:.4f}")
                botlog.info("BUY cost=%.2f px=%.6f filled=%.8f", target, px, filled)
                log_trade(now_iso(), "BUY", px, filled, "SIGNAL_BUY")
                time.sleep(SLEEP_SEC)
                continue

        # (optioneel) SELL-signaal bij cross omlaag -> alleen als netto winst
        if sig == "SELL" and position_amt > 0 and entry_price:
            net = net_change_pct(px, entry_price, FEE_PCT)
            if net >= MIN_PROFIT_PCT:
                sell_all_now(px, "SIGNAL_SELL")
                time.sleep(SLEEP_SEC)
                continue

        time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Gestopt.")
        break
    except Exception as e:
        print(f"[runtime] {e}")
        botlog.exception("runtime: %s", e)
        time.sleep(5)
