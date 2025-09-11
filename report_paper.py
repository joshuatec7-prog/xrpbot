# report_paper.py — leest /var/data/* en print een overzicht + DAGWINST
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import os
import sys

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
TRADES   = DATA_DIR / "trades.csv"
EQUITY   = DATA_DIR / "equity.csv"

def ts() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def _num(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series([None])

def print_daily_pnl():
    """Laat dagelijkse winst/verlies zien op basis van equity.csv."""
    if not EQUITY.exists():
        print("\n-- Dagelijkse winst --")
        print("(geen equity.csv gevonden; wordt automatisch gevuld door de bot)")
        return

    try:
        df = pd.read_csv(EQUITY)
    except Exception as e:
        print("\n-- Dagelijkse winst --")
        print(f"(equity.csv kon niet gelezen worden: {e})")
        return

    # Normaliseer kolommen
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols
    if "date" not in df.columns or "total_equity_eur" not in df.columns:
        print("\n-- Dagelijkse winst --")
        print("(equity.csv mist vereiste kolommen: date,total_equity_eur)")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["total_equity_eur"] = _num(df["total_equity_eur"])
    df = df.dropna(subset=["date", "total_equity_eur"]).sort_values("date")
    if df.empty:
        print("\n-- Dagelijkse winst --")
        print("(geen geldige regels in equity.csv)")
        return

    # Dagelijkse verandering
    df["pnl_day"] = df["total_equity_eur"].diff()

    # Laatste 10 dagen tonen
    print("\n-- Dagelijkse winst (laatste 10 dagen) --")
    tail = df.tail(10).copy()
    for _, r in tail.iterrows():
        d  = r["date"]
        eq = float(r["total_equity_eur"])
        dp = float(r["pnl_day"]) if pd.notna(r["pnl_day"]) else 0.0
        sign = "+" if dp >= 0 else ""
        print(f"{d} | equity=€{eq:,.2f} | dag={sign}€{dp:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    # Aggregaten (7d / 30d / YTD) — zolang de data beschikbaar is
    def sum_last_days(n: int) -> float:
        if len(df) < 2:
            return 0.0
        recent = df.tail(n+1)["pnl_day"].dropna()
        return float(recent.sum()) if not recent.empty else 0.0

    pnl_7d  = sum_last_days(7)
    pnl_30d = sum_last_days(30)

    print("\n-- Periode winst --")
    print(f"7d:  €{pnl_7d:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    print(f"30d: €{pnl_30d:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

def main():
    print(f"==> PAPER GRID RAPPORT ==  {ts()}")

    # 1) TRADES samenvatting
    if not TRADES.exists():
        print("(geen trades)")
    else:
        df = pd.read_csv(TRADES)

        # Normaliseer & numeriek
        for col in ("price","amount","fee_eur","realized_pnl_eur"):
            if col in df.columns:
                df[col] = _num(df[col])
        df["pair"] = df["pair"].astype(str)

        fees = float(_num(df.get("fee_eur", 0)).sum())
        pnl  = float(_num(df.get("realized_pnl_eur", 0)).sum())
        wins = int((df.get("realized_pnl_eur", pd.Series([])) > 0).sum())
        closes = int((df.get("side", pd.Series([])) == "SELL").sum())
        winrate = (wins / closes * 100) if closes else 0.0

        # Laatste equity uit equity.csv (indien aanwezig)
        latest_eq = "-"
        if EQUITY.exists():
            try:
                e = pd.read_csv(EQUITY)
                latest_eq = f"{float(_num(e['total_equity_eur']).iloc[-1]):.2f}"
            except Exception:
                pass

        print("\n-- Samenvatting --")
        print(f"Equity (laatst):  €{latest_eq}")
        print(f"Total fees:       €{fees:.2f}")
        print(f"Total realized:   €{pnl:.2f}")
        print(f"Winrate:          {winrate:.1f}% (wins={wins} / closes={closes})")

        # Per pair
        print("\n-- Per pair --")
        try:
            for pair, sub in df.groupby("pair"):
                f = float(_num(sub.get("fee_eur", 0)).sum())
                p = float(_num(sub.get("realized_pnl_eur", 0)).sum())
                buys  = int((sub.get("side", pd.Series([])) == "BUY").sum())
                sells = int((sub.get("side", pd.Series([])) == "SELL").sum())
                print(f"{pair:<7} | trades={len(sub):>3} | fees=€{f:>6.2f} | buys={buys} / sells={sells}")
        except Exception:
            pass

        # Laatste 10 trades
        print("\n-- Laatste 10 trades --")
        tail = df.tail(10).copy()
        for _, r in tail.iterrows():
            tm = r.get("timestamp", "")
            pair = r.get("pair", "")
            side = r.get("side", "")
            price = float(r.get("price", 0) or 0)
            amt   = float(r.get("amount", 0) or 0)
            rpnl  = float(r.get("realized_pnl_eur", 0) or 0)
            print(f"{tm} | {pair} | {side} | €{price:.6f} | {amt:.6f} | pnl=€{rpnl:.2f}")

    # 2) Dagelijkse winst uit equity.csv
    print_daily_pnl()

if __name__ == "__main__":
    main()
