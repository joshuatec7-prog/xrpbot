# main.py
import os
import time
import math
import ccxt
import logging
from datetime import datetime

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

def _mask(s: str | None) -> str:
    if not s:
        return "MISSING"
    return s[:4] + "..." + s[-4:]


# -----------------------------
# Env
# -----------------------------
API_KEY  = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
OPERATOR_ID = os.getenv("OPERATOR_ID")

SYMBOL = os.getenv("SYMBOL", "ETH/EUR")
BASE, QUOTE = SYMBOL.split("/")

LIVE = True   # deze worker is live

EUR_PER_TRADE = float(os.getenv("EUR_PER_TRADE", "15"))
MAX_EUR_PER_TRADE = float(os.getenv("MAX_EUR_PER_TRADE", "20"))
MAX_BUYS_PER_POSITION = int(os.getenv("MAX_BUYS_PER_POSITION", "1"))
ONE_POSITION_ONLY = os.getenv("ONE_POSITION_ONLY", "true").lower() == "true"

STOP_LOSS   = float(os.getenv("STOP_LOSS", "0.01"))   # 1% vaste SL
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.03")) # 3% vaste TP

TRAIL_TRIGGER = float(os.getenv("TRAIL_TRIGGER", "0.02"))  # activeer bij +2%
TRAIL_START   = float(os.getenv("TRAIL_START", "0.015"))   # eerste lock bij +1.5%
TRAIL_GAP     = float(os.getenv("TRAIL_GAP", "0.01"))      # trailing 1% onder piek

SLEEP_SEC = int(os.getenv("SLEEP_SEC", "60"))
TRADE_COOLDOWN_SEC = int(os.getenv("TRADE_COOLDOWN_SEC", "180"))  # na trade even afkoelen

REQUEST_INTERVAL_SEC = float(os.getenv("REQUEST_INTERVAL_SEC", "1.0"))

logging.info(f"[START] {datetime.now()} | LIVE={LIVE} | Symbol={SYMBOL}")
logging.info(f"API_KEY={_mask(API_KEY)} | API_SECRET={_mask(API_SECRET)} | operatorId={OPERATOR_ID}")


# -----------------------------
# Exchange
# -----------------------------
if not API_KEY or not API_SECRET:
    raise SystemExit("GEEN API SLEUTELS: zet API_KEY en API_SECRET in Secrets.")

ex = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
    "enableRateLimit": True,
})
# Zorg dat operatorId ALTIJD meegaat
ex.options["operatorId"] = OPERATOR_ID

markets = ex.load_markets()
if SYMBOL not in markets:
    raise SystemExit(f"{SYMBOL} niet gevonden op Bitvavo.")

market = markets[SYMBOL]
amount_prec = market.get("precision", {}).get("amount", None)
price_prec = market.get("precision", {}).get("price", None)
limits = market.get("limits", {}) or {}
MIN_BASE  = float((limits.get("amount") or {}).get("min") or 0.0)   # min base
MIN_QUOTE = float((limits.get("cost")   or {}).get("min") or 0.0)   # min EUR
logging.info(f"[INFO] Minima {SYMBOL}: min_cost=€{MIN_QUOTE}, min_amount={MIN_BASE}")

# -----------------------------
# Helpers
# -----------------------------
_last_call = 0.0
def throttle():
    global _last_call
    now = time.time()
    wait = REQUEST_INTERVAL_SEC - (now - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()

def amount_to_precision(amount: float) -> float:
    """Rond hoeveelheid naar exchange-precision af (neerwaarts)."""
    if amount_prec is None:
        return float(ex.amount_to_precision(SYMBOL, amount))
    q = 10 ** amount_prec
    return math.floor(amount * q) / q

def price_to_precision(price: float) -> float:
    if price_prec is None:
        return float(ex.price_to_precision(SYMBOL, price))
    q = 10 ** price_prec
    return math.floor(price * q) / q

def get_balance(asset: str) -> float:
    throttle()
    bal = ex.fetch_balance()
    return float((bal.get("free", {}) or {}).get(asset, 0.0))

def get_last_price() -> float:
    throttle()
    t = ex.fetch_ticker(SYMBOL)
    return float(t["last"])

def clamp_spend_eur(eur_free: float, px: float) -> float:
    """Bepaal spend in EUR incl. beurs-minima & CAP."""
    spend = min(eur_free, MAX_EUR_PER_TRADE, EUR_PER_TRADE)
    # moet voldoen aan min_cost OF min_amount
    need_eur = max(MIN_QUOTE, MIN_BASE * px, 0.50)
    if spend < need_eur:
        # probeer op te hogen tot need_eur binnen cap en saldo
        spend = min(max(need_eur, EUR_PER_TRADE), MAX_EUR_PER_TRADE, eur_free)
    return round(spend, 2)

def create_market_buy_by_eur(spend_eur: float):
    params = {"operatorId": OPERATOR_ID, "cost": spend_eur}
    throttle()
    return ex.create_order(SYMBOL, "market", "buy", None, None, params)

def create_market_sell_amount(amount_base: float):
    params = {"operatorId": OPERATOR_ID}
    throttle()
    return ex.create_order(SYMBOL, "market", "sell", amount_base, None, params)

# -----------------------------
# Trade-state
# -----------------------------
position_amt = 0.0           # hoeveelheid BASE die deze sessie heeft (opgebouwd)
entry_price: float | None = None
buys_count = 0
last_trade_ts = 0.0

# trailing
trail_active = False
trail_peak = None
trail_floor = None

def reset_trailing():
    global trail_active, trail_peak, trail_floor
    trail_active = False
    trail_peak = None
    trail_floor = None

def cooldown_active() -> bool:
    return (time.time() - last_trade_ts) < TRADE_COOLDOWN_SEC

# -----------------------------
# Strategie: eenvoudige MA cross?
# (we houden het nu bij “HOLD” → jij kunt hier later signaal inbouwen)
# Voor demo: we gebruiken een dummy: Koop als positie leeg; Verkoop op SL/TP/Trailing
# -----------------------------
def decide_signal(px: float) -> str:
    # hier zou je een echte strategie kunnen plakken (MA cross e.d.)
    # Voorbeeld: als we geen positie hebben -> BUY
    #            als wel positie -> HOLD (verkoop regelt SL/TP/Trailing)
    if position_amt <= 0.0:
        return "BUY"
    return "HOLD"


# -----------------------------
# Main loop
# -----------------------------
def run():
    global position_amt, entry_price, buys_count, last_trade_ts
    global trail_active, trail_peak, trail_floor

    logging.info(
        f"[PARAMS] €/trade={EUR_PER_TRADE} | CAP={MAX_EUR_PER_TRADE} "
        f"| ONE_POS={ONE_POSITION_ONLY} | MAX_BUYS={MAX_BUYS_PER_POSITION} "
        f"| SL={STOP_LOSS*100:.1f}% | TP={TAKE_PROFIT*100:.1f}% "
        f"| TRIGGER={TRAIL_TRIGGER*100:.1f}% | TSTART={TRAIL_START*100:.1f}% | TGAP={TRAIL_GAP*100:.1f}%"
    )

    while True:
        try:
            px = get_last_price()
            eur_free = get_balance(QUOTE)
            base_free = get_balance(BASE)

            # Voor de zekerheid: hanteer wat de bot denkt dat hij bezit (max op daadwerkelijke free)
            if position_amt > 0:
                position_amt = min(position_amt, base_free)

            logging.info(f"Prijs: €{px:.4f} | Signaal: {decide_signal(px)}")
            logging.debug(f"[DEBUG] Vrij {QUOTE}(bot): {eur_free:.2f} | Vrij {BASE}(bot): {base_free:.6f}")

            # ======== SL/TP + TRAILING ========
            if position_amt > 0 and entry_price:
                change = (px - entry_price) / entry_price

                # Trailing: triggeren?
                if not trail_active and change >= TRAIL_TRIGGER:
                    trail_active = True
                    trail_peak = px
                    start_lock = entry_price * (1 + TRAIL_START)
                    trail_floor = max(start_lock, trail_peak * (1 - TRAIL_GAP))
                    logging.info(f"[TRAIL] Activated. peak=€{trail_peak:.6f} floor=€{trail_floor:.6f}")

                if trail_active:
                    if px > trail_peak:
                        trail_peak = px
                        trail_floor = trail_peak * (1 - TRAIL_GAP)
                    if px <= trail_floor:
                        to_sell = amount_to_precision(position_amt)
                        if to_sell >= max(MIN_BASE, 0.0) and (to_sell * px) >= max(MIN_QUOTE, 0.5):
                            create_market_sell_amount(to_sell)
                            logging.info(f"[TRAIL] SELL {to_sell} {BASE} @ €{px:.6f} (peak=€{trail_peak:.6f})")
                            position_amt = 0.0
                            entry_price = None
                            buys_count = 0
                            last_trade_ts = time.time()
                            reset_trailing()
                            time.sleep(SLEEP_SEC)
                            continue

                # Vaste TP/SL
                if change >= TAKE_PROFIT or change <= -STOP_LOSS:
                    reason = "TP" if change >= TAKE_PROFIT else "SL"
                    to_sell = amount_to_precision(position_amt)
                    if to_sell >= max(MIN_BASE, 0.0) and (to_sell * px) >= max(MIN_QUOTE, 0.5):
                        create_market_sell_amount(to_sell)
                        logging.info(f"[LIVE] {reason} SELL {to_sell} {BASE} @ €{px:.6f} ({change*100:+.2f}%)")
                        position_amt = 0.0
                        entry_price = None
                        buys_count = 0
                        last_trade_ts = time.time()
                        reset_trailing()
                        time.sleep(SLEEP_SEC)
                        continue

            # ======== SELL signaal vanuit strategie? (optioneel) ========
            # In dit simpele voorbeeld doen we dat niet; verkoop komt via SL/TP/Trailing.

            # ======== BUY ========
            sig = decide_signal(px)
            if sig == "BUY":
                # position rules
                if ONE_POSITION_ONLY and position_amt > 0:
                    logging.info("[LIVE] ONE_POSITION_ONLY actief: reeds positie → geen BUY.")
                elif buys_count >= MAX_BUYS_PER_POSITION:
                    logging.info("[LIVE] MAX_BUYS_PER_POSITION bereikt → geen BUY.")
                elif cooldown_active():
                    left = TRADE_COOLDOWN_SEC - int(time.time() - last_trade_ts)
                    logging.info(f"[LIVE] In cooldown ({left}s) → geen BUY.")
                else:
                    spend = clamp_spend_eur(eur_free, px)
                    if spend < max(MIN_QUOTE, MIN_BASE * px, 0.50):
                        logging.info(f"[LIVE] BUY overslaan: bedrag te klein (nodig ≥ max({MIN_BASE:.6f} {BASE}, €{MIN_QUOTE:.2f})).")
                    else:
                        # bereken hoeveelheid en pas precision toe
                        est_amount = spend / px
                        amt = amount_to_precision(est_amount)
                        if amt < max(MIN_BASE, 0.0) or (amt * px) < max(MIN_QUOTE, 0.50):
                            logging.info("[LIVE] BUY overslaan (na precision/minima).")
                        else:
                            # op Bitvavo market-buy: fijner met 'cost' param (EUR)
                            order = create_market_buy_by_eur(spend)
                            # fallback filledAmount uit info
                            filled = None
                            try:
                                filled = float(order.get("info", {}).get("filledAmount"))
                            except Exception:
                                pass
                            if not filled:
                                filled = amt
                            position_amt += filled
                            entry_price = px  # reset entry op laatste buy
                            buys_count += 1
                            last_trade_ts = time.time()
                            reset_trailing()
                            logging.info(f"[LIVE] BUY €{spend:.2f} {BASE} (+{filled:.6f}) @ €{px:.6f}")

            time.sleep(SLEEP_SEC)

        except KeyboardInterrupt:
            logging.info("Gestopt door gebruiker.")
            break
        except Exception as e:
            logging.error(f"[ERROR] Runtime: {e!r}")
            time.sleep(3)


if __name__ == "__main__":
    run()
