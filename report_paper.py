# report_paper.py — leest /var/data/* en print overzicht met 'Sinds start' totaal
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import os
import sys

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
TRADES   = DATA_DIR / "trades.csv"
EQUITY   = DATA_DIR / "equity.csv"

def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def main():
    print(f"==> PAPER GRID RAPPORT ==  {ts()}")

    if not TRADES.exists():
        print("(geen trades)"); return

    # --- Data inlezen & normaliseren
    df = pd.read_csv(TRADES)
    # kolomnamen die we gebruiken: timestamp, side, fee_eur, realized_pnl_eur, price, amount, pair
    for col in ("price","amount","fee_eur","realized_pnl_eur"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = 0.0
    # timestamp -> date
    try:
        dt = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception:
        dt = pd.to_datetime(df.iloc[:,0], errors="coerce")  # fallback: 1e kolom
    df["date"] = dt.dt.tz_localize(None).dt.date

    # --- Aggregates
    fees = float(df["fee_eur"].sum())
    realized_total = float(df["realized_pnl_eur"].sum())  # <== Sinds start (totaal)
    wins  = int((df["realized_pnl_eur"] > 0).sum())
    closes = int((df["side"] == "SELL").sum())
    winrate = (wins / closes * 100.0) if closes else 0.0

    # Equity laatste waarde (optioneel)
    latest_eq = "-"
    if EQUITY.exists():
        try:
            e = pd.read_csv(EQUITY)
            latest_eq = f"€{float(e['total_equity_eur'].iloc[-1]):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            pass

    # Dag-tot-dag realized (alleen SELL-regels bevatten realized)
    daily = (
        df.groupby("date", dropna=True)["realized_pnl_eur"]
          .sum()
          .sort_index()
    )

    # 7d en 30d som (op basis van date-index)
    if not daily.empty:
        last_date = daily.index.max()
        d7  = float(daily.loc[[d for d in daily.index if (last_date - d).days < 7]].sum())
        d30 = float(daily.loc[[d for d in daily.index if (last_date - d).days < 30]].sum())
    else:
        d7 = d30 = 0.0

    # ====== Output ======
    print("-- Samenvatting --")
    print(f"Equity (laatst): {latest_eq}")
    print(f"Total fees: €{fees:.2f}")
    print(f"Totale realized (sinds start): €{realized_total:.2f}")
    print(f"Winrate:      {winrate:.1f}% (wins={wins} / closes={closes})")

    # Per pair overzicht
    if "pair" in df.columns:
        print("\n-- Per pair --")
        for pair, sub in df.groupby("pair"):
            f = float(sub["fee_eur"].sum())
            p = float(sub["realized_pnl_eur"].sum())
            buys  = int((sub["side"]=="BUY").sum())
            sells = int((sub["side"]=="SELL").sum())
            print(f"{pair:<8} | trades={len(sub):>3} | fees=€{f:>6.2f} | realized=€{p:>7.2f} | buys={buys} / sells={sells}")

    # Laatste 10 trades
    print("\n-- Laatste 10 trades --")
    tail = df.tail(10).copy()
    for _, r in tail.iterrows():
        dt_str = str(r.get("timestamp",""))[:19]
        pair   = r.get("pair","")
        side   = r.get("side","")
        price  = float(r.get("price",0) or 0)
        amt    = float(r.get("amount",0) or 0)
        pnl    = float(r.get("realized_pnl_eur",0) or 0)
        print(f"{dt_str} | {pair:>7} | {side:4} | €{price:>10.6f} | {amt:>10.6f} | pnl=€{pnl:>6.2f}")

    # Dagelijkse winst (zoals je gewend bent)
    print("\n-- Dagelijkse winst (laatste 10 dagen) --")
    if not daily.empty:
        last10 = daily.tail(10)
        for d, val in last10.items():
            print(f"{d} | equity=? | dag=€{val:.2f}")  # equity per dag blijft optioneel
    else:
        print("(geen realized data)")

    # Periode winst (7d/30d)
    print("\n-- Periode winst --")
    print(f"7d:  €{d7:.2f}")
    print(f"30d: €{d30:.2f}")

if __name__ == "__main__":
    main()
