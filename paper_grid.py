# report_paper.py — leest /var/data/* en print een overzicht
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import os, sys

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
TRADES   = DATA_DIR / "trades.csv"
EQUITY   = DATA_DIR / "equity.csv"
START_CAP = float(os.getenv("CAPITAL_EUR", "50000"))

def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def main():
    print(f"==> PAPER GRID RAPPORT ==  {ts()}")
    if not TRADES.exists():
        print("(geen trades)"); return

    df = pd.read_csv(TRADES)
    for col in ("price","amount","fee_eur","realized_pnl_eur"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    fees   = df["fee_eur"].sum()
    pnl    = df["realized_pnl_eur"].sum()
    wins   = (df["realized_pnl_eur"] > 0).sum()
    closes = (df["side"] == "SELL").sum()
    winrate = (wins / closes * 100) if closes else 0.0

    print("-- Samenvatting --")
    latest_eq_str = "-"
    if EQUITY.exists():
        try:
            e = pd.read_csv(EQUITY)
            latest_eq_str = f"{float(e['total_equity_eur'].iloc[-1]):.2f}"
        except: pass

    print(f"Equity (laatst): €{latest_eq_str}")
    try:
        le = float(latest_eq_str)
        print(f"Winst vs start (€{START_CAP:,.2f}): €{(le-START_CAP):.2f}".replace(",", "X").replace(".", ",").replace("X","."))
    except:
        pass

    print(f"Totale fees: €{fees:.2f}")
    print(f"Totaal realized: €{pnl:.2f}")
    print(f"Winrate:     {winrate:.1f}% (wins={wins} / closes={closes})")

    print("\n-- Per pair --")
    for pair, sub in df.groupby("pair"):
        f = sub["fee_eur"].sum()
        p = sub["realized_pnl_eur"].sum()
        buys  = (sub["side"]=="BUY").sum()
        sells = (sub["side"]=="SELL").sum()
        print(f"{pair:<7} | trades={len(sub):>3} | fees=€{f:>6.2f} | pnl=€{p:>7.2f} | buys={buys} / sells={sells}")

    print("\n-- Laatste 10 trades --")
    tail = df.tail(10).copy()
    for _, r in tail.iterrows():
        print(f"{r['timestamp']} | {r['pair']} | {r['side']} | €{r['price']:.6f} | {r['amount']:.6f} | pnl=€{r['realized_pnl_eur']:.2f}")

if __name__ == "__main__":
    main()
