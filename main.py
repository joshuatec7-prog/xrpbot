import os
import time
import logging
import ccxt

# =====================
# CONFIG & ENVIRONMENT
# =====================
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
OPERATOR_ID = os.getenv("OPERATOR_ID")
SYMBOL = os.getenv("SYMBOL", "ETH/EUR")

EUR_PER_TRADE = float(os.getenv("EUR_PER_TRADE", 15))
MAX_EUR_PER_TRADE = float(os.getenv("MAX_EUR_PER_TRADE", 20))
TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", 0.03))  # 3%
STOP_LOSS = float(os.getenv("STOP_LOSS", 0.01))      # 1%
SLEEP_SEC = int(os.getenv("SLEEP_SEC", 60))

# =====================
# LOGGING
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# =====================
# INIT EXCHANGE
# =====================
exchange = ccxt.bitvavo({
    "apiKey": API_KEY,
    "secret": API_SECRET,
})
exchange.options["operatorId"] = OPERATOR_ID

logging.info(f"Bot gestart | Symbol={SYMBOL} | EUR_PER_TRADE={EUR_PER_TRADE} | TAKE_PROFIT={TAKE_PROFIT*100}% | STOP_LOSS={STOP_LOSS*100}%")


# =====================
# MAIN LOOP
# =====================
def run_bot():
    last_buy_price = None
    amount = 0

    while True:
        try:
            # Haal laatste prijs
            ticker = exchange.fetch_ticker(SYMBOL)
            price = ticker["last"]
            logging.info(f"Prijs: €{price} | Signaal check...")

            # Als we nog niets gekocht hebben → KOPEN
            if last_buy_price is None:
                params = {"operatorId": OPERATOR_ID, "cost": EUR_PER_TRADE}
                order = exchange.create_order(SYMBOL, "market", "buy", None, None, params)
                last_buy_price = price
                amount = float(order["filled"])
                logging.info(f"[LIVE] BUY @ €{price} | Hoeveelheid: {amount} {SYMBOL.split('/')[0]}")

            else:
                # TAKE PROFIT
                if price >= last_buy_price * (1 + TAKE_PROFIT):
                    params = {"operatorId": OPERATOR_ID}
                    order = exchange.create_order(SYMBOL, "market", "sell", amount, None, params)
                    logging.info(f"[LIVE] SELL (TP) @ €{price} | Winst: {(price-last_buy_price):.2f} EUR per coin")
                    last_buy_price = None
                    amount = 0

                # STOP LOSS
                elif price <= last_buy_price * (1 - STOP_LOSS):
                    params = {"operatorId": OPERATOR_ID}
                    order = exchange.create_order(SYMBOL, "market", "sell", amount, None, params)
                    logging.warning(f"[LIVE] SELL (SL) @ €{price} | Verlies: {(last_buy_price-price):.2f} EUR per coin")
                    last_buy_price = None
                    amount = 0

        except Exception as e:
            logging.error(f"Runtime error: {e}")

        time.sleep(SLEEP_SEC)


if __name__ == "__main__":
    run_bot()
