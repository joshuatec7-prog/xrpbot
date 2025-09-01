# report_paper.py
# 12-uurlijkse/handmatige rapportage voor de paper grid bot
# - Leest /var/data/{trades.csv,equity.csv,state.json}
# - Haalt actuele prijzen op (Bitvavo public) voor mark-to-market
# - Toont ALLE paren uit ENV COINS (ook zonder trades)

import os, json, time, csv
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

# ---------------- ENV & paden ----------------
DATA_DIR     = Path(os.getenv("DATA_DIR", "/var/data"))
TRADES_CSV   = DATA_DIR / "trades.csv"
EQUITY_CSV   = DATA_DIR / "equity.csv"
STATE_FILE   = DATA_DIR / "state.json"

COINS = [x.strip().upper() for x in os.getenv(
    "COINS",
    "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
).split(",") if x.strip()]

CAPITAL_EUR  = float(os.getenv("CAPITAL_EUR", "15000"))
EXCHANGE_ID  = os.getenv("EXCHANGE", "bitvavo")

# --------------- helpers ---------------------
def ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

def load_json_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def to_num(s, default=0.0):
    try:
        return float(s)
    except Exception:
        return default

# --------------- prijzen ophalen (public) ---------------
def fetch_prices(pairs):
    try:
        ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})
        ex.load_markets()
        px = {}
        for p in pairs:
            if p in ex.markets:
                try:
                    px[p] = float(ex.fetch_ticker(p)["last"])
                except Exception:
                    px[p] = None
            else:
                px[p] = None
        return px
    except Exception:
        return {p: None for p in pairs}

# --------------- main ---------------
def main():
    trades = load_csv_safe(TRADES_CSV)
    equity_snap = load_csv_safe(EQUITY_CSV)
    state = load_json_safe(STATE_FILE)

    # Zorg dat kolommen numeriek zijn als ze bestaan
    if not trades.empty:
        for col in ("price", "amount", "fee_eur", "realized_pnl_eur"):
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce").fillna(0.0)
    else:
        trades = pd.DataFrame(columns=[
            "timestamp","pair","side","price","amount","fee_eur",
            "cash_eur","coin","coin_qty","realized_pnl_eur","comment"
        ])

    # Samenvatting totaal
    total_fees = trades["fee_eur"].sum() if "fee_eur" in trades else 0.0
    realized   = trades["realized_pnl_eur"].sum() if "realized_pnl_eur" in trades else 0.0

    # Winrate (alleen SELL trades met realized_pnl_eur > 0)
    wins = 0
    total_closes = 0
    if not trades.empty:
        sells = trades[trades["side"].str.upper() == "SELL"] if "side" in trades else pd.DataFrame()
        if not sells.empty and "realized_pnl_eur" in sells:
            wins = (sells["realized_pnl_eur"] > 0).sum()
            total_closes = len(sells)
    winrate = (wins / total_closes * 100.0) if total_closes else 0.0

    # Actuele coin posities uit state
    cash = 0.0
    coin_qty = {p: 0.0 for p in COINS}
    if state:
        cash = float(state.get("portfolio", {}).get("portfolio_eur", 0.0))
        for p in COINS:
            try:
                coin_qty[p] = float(state.get("portfolio", {}).get("coins", {}).get(p, {}).get("qty", 0.0))
            except Exception:
                coin_qty[p] = 0.0

    # Huidige prijzen
    prices = fetch_prices(COINS)
    coin_value = 0.0
    per_pair_value = {}
    for p in COINS:
        px = prices.get(p)
        val = (coin_qty[p] * px) if (px and coin_qty[p] > 0) else 0.0
        per_pair_value[p] = val
        coin_value += val

    equity_live = cash + coin_value if (cash or coin_value) else None
    equity_fallback = None
    if (equity_live is None) and not equity_snap.empty:
        try:
            equity_fallback = float(pd.to_numeric(equity_snap["total_equity_eur"], errors="coerce").dropna().iloc[-1])
        except Exception:
            equity_fallback = None

    equity_show = equity_live if equity_live is not None else equity_fallback if equity_fallback is not None else CAPITAL_EUR

    # Per pair statistieken uit trades
    per_pair_stats = {p: {"trades": 0, "pnl": 0.0, "fees": 0.0} for p in COINS}
    if not trades.empty:
        for p in COINS:
            dfp = trades[trades["pair"] == p] if "pair" in trades else pd.DataFrame()
            per_pair_stats[p]["trades"] = len(dfp)
            if "realized_pnl_eur" in dfp:
                per_pair_stats[p]["pnl"] = dfp["realized_pnl_eur"].sum()
            if "fee_eur" in dfp:
                per_pair_stats[p]["fees"] = dfp["fee_eur"].sum()

    # ----------- Print rapport -----------
    print(f"==> PAPER GRID RAPPORT ==  {ts()}")
    print()
    print("-- Samenvatting --")
    print(f"Startkapitaal : €{CAPITAL_EUR:,.2f}")
    print(f"Actuele equity: €{equity_show:,.2f}" + (" (live)" if equity_live is not None else " (fallback)"))
    print(f"EUR cash      : €{cash:,.2f}")
    print(f"Coin-waarde   : €{coin_value:,.2f}")
    print(f"Fees (som)    : €{total_fees:,.2f}")
    print(f"Gerealiseerd  : €{realized:,.2f}")
    print(f"Winrate       : {winrate:.1f}%  (wins={wins} / closes={total_closes})")
    nett = equity_show - CAPITAL_EUR
    print(f"==> Netto resultaat (totaal): €{nett:,.2f}")
    print()

    print("-- Per pair --")
    for p in COINS:
        px = prices.get(p)
        px_str = f"€{px:,.6f}" if px else "n/a"
        print(f"{p:7s} | trades={per_pair_stats[p]['trades']:3d} | pnl=€{per_pair_stats[p]['pnl']:>9.2f} | "
              f"fees=€{per_pair_stats[p]['fees']:>7.2f} | qty={coin_qty[p]:.6f} | px={px_str} | value=€{per_pair_value[p]:,.2f}")
    print()

    # Laatste trades
    print("-- Laatste 10 trades --")
    if trades.empty:
        print("(geen trades)")
    else:
        cols = ["timestamp","pair","side","price","amount","realized_pnl_eur"]
        for c in cols:
            if c not in trades.columns:
                trades[c] = ""
        last = trades.tail(10)
        for _, r in last.iterrows():
            ts_ = str(r["timestamp"])
            pair = str(r["pair"])
            side = str(r["side"])
            price = to_num(r["price"])
            amount = to_num(r["amount"])
            rpnl = to_num(r["realized_pnl_eur"])
            print(f"{ts_} | {pair} | {side} | {price:,.6f} | {amount:.6f} | pnl=€{rpnl:.2f}")

if __name__ == "__main__":
    main()
