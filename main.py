# ==============================
#  main.py  —  Bitvavo MA-cross bot (Render-ready)
# ==============================
# Belangrijkste fixes:
# - Unieke clientOrderId per order (Bitvavo errorCode 205 opgelost)
# - Geen dubbele buys (cooldown + ONE_POSITION_ONLY)
# - Market BUY met 'cost', SELL met 'amount'
# - Respecteert min_cost & min_amount (Bitvavo minima)
# ==============================

import os
import time
import csv
import uuid
import random
import pathlib
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import ccxt

# --------------------------------------------------
# Optioneel keep-alive voor pingdiensten (mag ontbreken)
# --------------------------------------------------
try:
    from keep_alive import keep_alive  # simpele Flask die "/" serveert
    keep_alive()
except Exception:
    pass

# --------------------------------------------------
# Helpers
# --------------------------------------------------
TZ = ZoneInfo("Europe/Amsterdam")
def now_iso():
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

def mask(s: str | None) -> str:
    if not s:
        return "MISSING"
    return s[:4] + "..." + s[-4:]

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def new_client_id(prefix: str) -> str:
    # Unieke (korte) id die Bitvavo accepteert
    return f"{prefix}-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"

# --------------------------------------------------
# Config uit ENV
# --------------------------------------------------
API_KEY  = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")

if not API_KEY or not API_SECRET:
    raise SystemExit("Geen API_KEY/API_SECRET gevonden in environment variables.")

SYMBOL     = os.environ.get("SYMBOL", "ETH/EUR").strip()
TIMEFRAME  = os.environ.get("TIMEFRAME", "1m").strip()
LIVE       = os.environ.get("LIVE", "true").lower() in ("1", "true", "yes")

FAST = int(os.environ.get("FAST", "3"))
SLOW = int(os.environ.get("SLOW", "6"))

EUR_PER_TRADE     = float(os.environ.get("EUR_PER_TRADE", "10"))
MAX_EUR_PER_TRADE = float(os.environ.get("MAX_EUR_PER_TRADE", "20"))

SLEEP_SEC            = float(os.environ.get("SLEEP_SEC", "60"))
COOLDOWN_SEC         = float(os.environ.get("COOLDOWN_SEC", "75"))
TRADE_COOLDOWN_SEC   = float(os.environ.get("TRADE_COOLDOWN_SEC", "180"))

ONE_POSITION_ONLY        = os.environ.get("ONE_POSITION_ONLY", "true").lower() in ("1","true","yes")
LOCK_PREEXISTING_BALANCE = os.environ.get("LOCK_PREEXISTING_BALANCE", "false").lower() in ("1","true","yes")

ORDER_ID_PREFIX = (os.environ.get("ORDER_ID_PREFIX", "bot")).strip() or "bot"

# --------------------------------------------------
# Logging
# --------------------------------------------------
LOG_DIR = pathlib.Path(".")
LOG_DIR.mkdir(exist_ok=True)
TRADES_CSV = LOG_DIR / "trades.csv"

def log_trade(mode: str, side: str, price: float, amount_or_cost: float, note: str):
    new_file = not TRADES_CSV.exists()
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp","mode","side","price","amount_or_cost","note"])
        w.writerow([now_iso(), mode, side, f"{price:.8f}", f"{amount_or_cost:.8f}", note])

# --------------------------------------------------
# Exchange init
# --------------------------------------------------
ex = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
})

# Market buy via cost
ex.options["createMarketBuyOrderRequiresPrice"] = False

# Laad markets & minima
markets = ex.load_markets()
if SYMBOL not in markets:
    raise SystemExit(f"Symbool {SYMBOL} niet gevonden op Bitvavo.")

BASE, QUOTE = SYMBOL.split("/")
limits = markets[SYMBOL].get("limits", {}) or {}
MIN_BASE  = float((limits.get("amount") or {}).get("min") or 0.0)   # minimale base-amount
MIN_COST  = float((limits.get("cost")   or {}).get("min") or 0.0)   # minimale quote-kost (EUR)

print(f"[START] {now_iso()}  | LIVE={LIVE} | Symbol={SYMBOL} | TF={TIMEFRAME}")
print(f"[INFO ] Minima: min_cost=€{MIN_COST}, min_amount={MIN_BASE} {BASE}")
print(f"[INFO ] EUR_PER_TRADE=€{EUR_PER_TRADE} | MAX_EUR_PER_TRADE=€{MAX_EUR_PER_TRADE}")
print(f"[INFO ] ONE_POSITION_ONLY={ONE_POSITION_ONLY} | LOCK_PREEXISTING_BALANCE={LOCK_PREEXISTING_BALANCE}")
print(f"[INFO ] API_KEY={mask(API_KEY)} | API_SECRET={mask(API_SECRET)}")

# --------------------------------------------------
# Retry & rate-limit helper (simpele variant)
# --------------------------------------------------
def with_retry(func, *args, **kwargs):
    delay = 1.0
    for attempt in range(1, 6):
        try:
            return func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout, ccxt.RateLimitExceeded) as e:
            time.sleep(delay + random.uniform(0, 0.4))
            delay *= 1.5
        except Exception as e:
            # Laat andere fouten meteen zien (zoals bad params)
            raise
    raise RuntimeError("Max retries reached for exchange call.")

# --------------------------------------------------
# Data & signaal
# --------------------------------------------------
def fetch_ohlcv(limit: int = 60) -> pd.DataFrame:
    raw = with_retry(ex.fetch_ohlcv, SYMBOL, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(raw, columns=["t","o","h","l","c","v"])
    return df

def make_signal(df: pd.DataFrame) -> str:
    if len(df) < max(FAST, SLOW) + 3:
        return "HOLD"
    fast = df["c"].rolling(FAST).mean()
    slow = df["c"].rolling(SLOW).mean()
    # kruis op de voorlaatste candle
    if fast.iloc[-2] > slow.iloc[-2] and fast.iloc[-3] <= slow.iloc[-3]:
        return "BUY"
    if fast.iloc[-2] < slow.iloc[-2] and fast.iloc[-3] >= slow.iloc[-3]:
        return "SELL"
    return "HOLD"

# --------------------------------------------------
# Startbalans (voor lock-preexisting)
# --------------------------------------------------
all_bal = with_retry(ex.fetch_balance)
start_free_eur  = float(all_bal.get("free", {}).get(QUOTE, 0) or 0.0)
start_free_base = float(all_bal.get("free", {}).get(BASE, 0)  or 0.0)

# --------------------------------------------------
# Positie en cooldowns
# --------------------------------------------------
position_base = 0.0        # door deze sessie opgebouwde BASE
entry_price   = None        # gemiddelde instapprijs
last_buy_ts   = 0.0
last_sell_ts  = 0.0

EPS = 1e-8

# --------------------------------------------------
# Kern helpers voor orders
# --------------------------------------------------
def buy_market_by_cost(eur_cost: float, price_hint: float):
    """
    Market BUY via 'cost'. Bitvavo verwacht param 'cost' in EUR.
    """
    if eur_cost < max(MIN_COST, 0.5):
        print(f"[LIVE] BUY overslaan: cost €{eur_cost:.2f} < min €{max(MIN_COST,0.5):.2f}")
        return None

    params = {
        "cost": float(f"{eur_cost:.2f}"),              # nette afronding
        "clientOrderId": new_client_id(ORDER_ID_PREFIX)
    }
    order = with_retry(ex.create_order, SYMBOL, "market", "buy", None, None, params)
    return order

def sell_market_by_amount(amount: float):
    """
    Market SELL via amount in BASE.
    """
    if amount < max(MIN_BASE, 0.000001):
        print(f"[LIVE] SELL overslaan: amount {amount:.8f} < min {max(MIN_BASE, 0.000001):.8f}")
        return None

    params = {
        "clientOrderId": new_client_id(ORDER_ID_PREFIX)
    }
    order = with_retry(ex.create_order, SYMBOL, "market", "sell", amount, None, params)
    return order

# --------------------------------------------------
# Main loop
# --------------------------------------------------
while True:
    try:
        df = fetch_ohlcv(limit=max(100, SLOW + 5))
        price = float(df["c"].iloc[-1])
        sig   = make_signal(df)

        # Balances
        bal = with_retry(ex.fetch_balance)
        free_eur  = float(bal.get("free", {}).get(QUOTE, 0) or 0.0)
        free_base = float(bal.get("free", {}).get(BASE,  0) or 0.0)

        if LOCK_PREEXISTING_BALANCE:
            eur_room  = max(0.0, free_eur  - start_free_eur)
            base_room = max(0.0, free_base - start_free_base)
        else:
            eur_room  = free_eur
            base_room = free_base

        print(f"Prijs: €{price:.4f} | Signaal: {sig}")
        print(f"[DEBUG] Vrij EUR(bot): {eur_room:.2f} | Vrij {BASE}(bot): {base_room:.8f}")

        now = time.time()

        # ---------------- BUY ----------------
        if sig == "BUY":
            # cooldown & single position
            if now - last_buy_ts < COOLDOWN_SEC:
                print("[DBG] Buy cooldown actief.")
            elif ONE_POSITION_ONLY and (position_base > EPS):
                print("[DBG] ONE_POSITION_ONLY actief: al positie.")
            else:
                # Bepaal te besteden bedrag
                target = clamp(EUR_PER_TRADE, 0, MAX_EUR_PER_TRADE)
                target = min(target, eur_room)

                # check minima (en corrigeer indien nodig binnen cap en room)
                min_required = max(MIN_COST, price * MIN_BASE, 0.50)
                if target + 1e-9 < min_required:
                    target = min(max(min_required, EUR_PER_TRADE), MAX_EUR_PER_TRADE, eur_room)

                if target + 1e-9 < min_required:
                    print(f"[LIVE] BUY overslaan: te weinig EUR (nodig ≥ €{min_required:.2f}).")
                else:
                    if LIVE:
                        order = buy_market_by_cost(target, price)
                        if order:
                            # filled base approx:
                            filled = None
                            try:
                                filled = float(order.get("info", {}).get("filledAmount") or 0.0)
                            except Exception:
                                pass
                            if not filled or filled <= 0:
                                filled = target / price

                            position_base += filled
                            entry_price = price if entry_price is None else entry_price
                            last_buy_ts = now

                            print(f"[LIVE] BUY €{target:.2f} ≈ +{filled:.6f} {BASE} @ €{price:.4f}")
                            log_trade("LIVE", "BUY", price, target, "signal")
                    else:
                        # paper
                        est_filled = target / price
                        position_base += est_filled
                        entry_price = price if entry_price is None else entry_price
                        last_buy_ts = now
                        print(f"[PAPER] BUY €{target:.2f} ≈ +{est_filled:.6f} {BASE} @ €{price:.4f}")
                        log_trade("PAPER", "BUY", price, target, "signal")

        # ---------------- SELL ----------------
        elif sig == "SELL":
            # cooldown
            if now - last_sell_ts < TRADE_COOLDOWN_SEC:
                print("[DBG] Sell cooldown actief.")
            else:
                # verkoop maximaal wat bij bot hoort (position_base) én beschikbaar is (base_room)
                to_sell = min(position_base, base_room)
                if to_sell > EPS:
                    if LIVE:
                        order = sell_market_by_amount(to_sell)
                        if order:
                            position_base = max(0.0, position_base - to_sell)
                            last_sell_ts = now
                            print(f"[LIVE] SELL {to_sell:.6f} {BASE} @ ~€{price:.4f}")
                            log_trade("LIVE", "SELL", price, to_sell, "signal")
                            if position_base <= EPS:
                                entry_price = None
                    else:
                        position_base = max(0.0, position_base - to_sell)
                        last_sell_ts = now
                        print(f"[PAPER] SELL {to_sell:.6f} {BASE} @ €{price:.4f}")
                        log_trade("PAPER", "SELL", price, to_sell, "signal")
                else:
                    print("[DBG] Niets van deze sessie om te verkopen.")

        time.sleep(SLEEP_SEC)

    except KeyboardInterrupt:
        print("Gestopt door gebruiker.")
        break
    except Exception as e:
        # Laat duidelijk zien welke fout er is
        print(f"[LIVE] ❌ Runtime error: {e!r}")
        time.sleep(3)
