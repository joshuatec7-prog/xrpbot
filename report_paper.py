# report_paper.py — leest /var/data/* en print een overzicht
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import os
import sys
import csv

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
TRADES   = DATA_DIR / "trades.csv"
EQUITY   = DATA_DIR / "equity.csv"
DAILY    = DATA_DIR / "daily_pnl.csv"   # <— NIEUW

def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def main():
    print(f"==> PAPER GRID RAPPORT ==  {ts()}")

    if not TRADES.exists():
        print("(geen trades)")
        # laat alsnog daily pnl zien als die er is
        show_daily()
        return

    df = pd.read_csv(TRADES)
    df["price"]   = pd.to_numeric(df["price"], errors="coerce")
    df["amount"]  = pd.to_numeric(df["amount"], errors="coerce")
    df["fee_eur"] = pd.to_numeric(df["fee_eur"], errors="coerce")
    df["realized_pnl_eur"] = pd.to_numeric(df["realized_pnl_eur"], errors="coerce")

    fees = df["fee_eur"].sum()
    pnl  = df["realized_pnl_eur"].sum()
    wins = (df["realized_pnl_eur"] > 0).sum()
    closes = (df["side"] == "SELL").sum()
    winrate = (wins / closes * 100) if closes else 0.0

    print("-- Samenvatting --")
    latest_eq = "-"
    if EQUITY.exists():
        try:
            e = pd.read_csv(EQUITY)
            latest_eq = f"€{float(e['total_equity_eur'].iloc[-1]):.2f}"
        except Exception:
            pass
    print(f"Equity (laatst): {latest_eq}")
    print(f"Total fees:      €{fees:.2f}")
    print(f"Total realized:  €{pnl:.2f}")
    print(f"Winrate:         {winrate:.1f}% (wins={wins} / closes={closes})")

    print("\n-- Per pair --")
    for pair, sub in df.groupby("pair"):
        f = sub["fee_eur"].sum()
        p = sub["realized_pnl_eur"].sum()
        buys  = (sub["side"]=="BUY").sum()
        sells = (sub["side"]=="SELL").sum()
        print(f"{pair:<7} | trades={len(sub):>3} | long=€{p:>6.2f} | fees=€{f:>6.2f} | buys={buys} / sells={sells}")

    print("\n-- Laatste 10 trades --")
    tail = df.tail(10).copy()
    for _, r in tail.iterrows():
        print(f"{r['timestamp']} | {r['pair']} | {r['side']} | €{r['price']:.6f} | {r['amount']:.6f} | pnl=€{r['realized_pnl_eur']:.2f}")

    # NIEUW: dagwinst onderaan
    show_daily()

def show_daily():
    print("\n-- Dagwinst (laatste 7 dagen) --")
    if DAILY.exists():
        try:
            d = pd.read_csv(DAILY)
            d = d.tail(7)
            for _, r in d.iterrows():
                eq  = float(r['equity_eur'])
                day = float(r['day_pnl_eur'])
                cum = float(r['cum_pnl_eur'])
                print(f"{r['date']} | equity=€{eq:.2f} | day=€{day:+.2f} | cum=€{cum:+.2f}")
        except Exception as e:
            print(f"(kon daily_pnl.csv niet lezen: {e})")
    else:
        print("(nog geen daily_pnl.csv)")

if __name__ == "__main__":
    main()
