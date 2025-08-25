import os
import time
import logging
from datetime import datetime, timedelta

import ccxt


# =========================
# Omgevingsvariabelen
# =========================
API_KEY     = os.getenv("API_KEY", "")
API_SECRET  = os.getenv("API_SECRET", "")
OPERATOR_ID = os.getenv("OPERATOR_ID", "")          # VERPLICHT op Bitvavo
SYMBOL      = os.getenv("SYMBOL", "ETH/EUR")

EUR_PER_TRADE      = float(os.getenv("EUR_PER_TRADE", "15"))
MAX_EUR_PER_TRADE  = float(os.getenv("MAX_EUR_PER_TRADE", "20"))

TAKE_PROFIT        = float(os.getenv("TAKE_PROFIT", "0.03"))  # 3% winst
STOP_LOSS          = float(os.getenv("STOP_LOSS", "0.01"))    # 1% verlies

SLEEP_SEC          = int(float(os.getenv("SLEEP_SEC", "60")))      # wachttijd per loop
TRADE_COOLDOWN_SEC = int(float(os.getenv("TRADE_COOLDOWN_SEC", "180")))  # pauze na verkoop

# Fallback voor minimum-orderbedrag als de exchange-limieten niets geven
MIN_QUOTE_FALLBACK = 5.0


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def assert_env():
    if not API_KEY or not API_SECRET:
        raise SystemExit("GEEN API sleutels: zet API_KEY en API_SECRET in de Environment Variables.")
    if not OPERATOR_ID:
        raise SystemExit("OPERATOR_ID ontbreekt: voeg OPERATOR_ID toe aan de Environment Variables.")


# =========================
# Exchange setup
# =========================
def init_exchange():
    ex = ccxt.bitvavo({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })
    # Bitvavo accepteert 'cost' bij market buy (geen price nodig)
    ex.options["createMarketBuyOrderRequiresPrice"] = False
    # Operator ID verplicht:
    ex.options["operatorId"] = OPERATOR_ID
    return ex


# =========================
# Helpers
# =========================
def amount_to_precision(ex, symbol, amount):
    return float(ex.amount_to_precision(symbol, amount))

def quote_to_precision(ex, symbol, cost):
    # Bitvavo heeft geen "cost_to_precision", maar prijs afronden hoeft niet bij 'cost'.
    return round(cost, 2)


# =========================
# Bot logica
# =========================
def run():
    assert_env()
    ex = init_exchange()

    # Markten + limieten (minima)
    markets = ex.load_markets()
    if SYMBOL not in markets:
        raise SystemExit(f"Symbool {SYMBOL} niet gevonden op Bitvavo.")

    limits = markets[SYMBOL].get("limits", {})
    min_amount = float((limits.get("amount") or {}).get("min") or 0.0)
    min_cost   = float((limits.get("cost")   or {}).get("min") or MIN_QUOTE_FALLBACK)

    base, quote = SYMBOL.split("/")
    logging.info(
        f"START | {SYMBOL} | €/trade={EUR_PER_TRADE} (max {MAX_EUR_PER_TRADE}) | "
        f"TP={TAKE_PROFIT*100:.1f}% | SL={STOP_LOSS*100:.1f}% | min_cost=€{min_cost}, min_amount={min_amount} {base}"
    )

    # Bot state
    position_size = 0.0       # hoeveelheid BASE in positie
    entry_price   = None      # instapprijs
    last_sell_at  = None      # cooldown timer start

    while True:
        try:
            # 1) huidige prijs
            ticker = ex.fetch_ticker(SYMBOL)
            price = float(ticker["last"] or ticker["close"] or ticker["ask"] or ticker["bid"])
            logging.info(f"Prijs: €{price:.4f} | State: {'LONG' if position_size>0 else 'FLAT'}")

            # 2) Als FLAT -> eventueel kopen (met cooldown)
            if position_size <= 0:
                # respecteer cooldown na verkoop
                if last_sell_at and datetime.utcnow() < (last_sell_at + timedelta(seconds=TRADE_COOLDOWN_SEC)):
                    remain = (last_sell_at + timedelta(seconds=TRADE_COOLDOWN_SEC)) - datetime.utcnow()
                    logging.info(f"Cooldown actief nog ~{int(remain.total_seconds())}s. Geen nieuwe koop.")
                else:
                    # bedrag bepalen met beurzen-minima en gebruiker-caps
                    target_cost = max(EUR_PER_TRADE, min_cost)
                    target_cost = min(target_cost, MAX_EUR_PER_TRADE)
                    target_cost = quote_to_precision(ex, SYMBOL, target_cost)

                    est_base = target_cost / price
                    if min_amount > 0 and est_base < min_amount:
                        logging.info(
                            f"Koop overslaan: te weinig amount ({est_base:.6f}<{min_amount}) bij €{target_cost:.2f}."
                        )
                    else:
                        params = {"operatorId": OPERATOR_ID, "cost": target_cost}
                        order = ex.create_order(SYMBOL, "market", "buy", None, None, params)

                        # filled hoeveelheid (BASE)
                        filled = None
                        try:
                            filled = float(order.get("info", {}).get("filledAmount"))
                        except Exception:
                            pass
                        if not filled:
                            filled = target_cost / price

                        position_size = amount_to_precision(ex, SYMBOL, filled)
                        entry_price = price
                        logging.info(f"[LIVE] BUY ~€{target_cost:.2f} | +{position_size:.6f} {base} @ €{entry_price:.4f}")

            # 3) Als LONG -> TP / SL checken en verkopen
            else:
                change = (price - entry_price) / entry_price
                if change >= TAKE_PROFIT or change <= -STOP_LOSS:
                    reason = "TP" if change >= TAKE_PROFIT else "SL"
                    qty = amount_to_precision(ex, SYMBOL, position_size)

                    if qty > 0:
                        params = {"operatorId": OPERATOR_ID}
                        ex.create_order(SYMBOL, "market", "sell", qty, None, params)
                        logging.info(
                            f"[LIVE] SELL {reason} | {qty:.6f} {base} @ €{price:.4f} | P&L={change*100:+.2f}%"
                        )
                        # reset state & cooldown starten
                        position_size = 0.0
                        entry_price = None
                        last_sell_at = datetime.utcnow()

            time.sleep(SLEEP_SEC)

        except Exception as e:
            logging.error(f"Runtime error: {repr(e)}")
            time.sleep(5)


if __name__ == "__main__":
    run()
