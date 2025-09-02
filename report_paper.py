# report_paper.py
# 12-uurlijkse rapportage voor multi-coin paper grid (long + short)
# - Leest trades.csv & equity.csv uit DATA_DIR (default /var/data)
# - Telt fees, realized PnL, per-pair breakdown (long+short)
# - Print overzicht naar logs

import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
TRADES = DATA_DIR / "trades.csv"
EQUITY = DATA_DIR / "equity.csv"

def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def safe_read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def money(x) -> str:
    try:
        return f"€{float(x):,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return "€0,00"

def main():
    df = safe_read_csv(TRADES)
    eq = safe_read_csv(EQUITY)

    print(f"==> PAPER GRID RAPPORT ==  {ts()}")

    if df.empty:
        print("(geen trades)")
        # Toon equity snapshot als die er is
        if not eq.empty:
            last_row = eq.tail(1).iloc[0]
            print(f"Equity (laatst): {money(last_row.get('total_equity_eur', 0))} (op {last_row.get('date')})")
        return

    # Zorg dat de numerieke kolommen echt numeriek zijn
    for col in ["price","amount","fee_eur","cash_eur","realized_pnl_eur"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Totals
    total_fees = df["fee_eur"].sum() if "fee_eur" in df else 0.0
    total_realized = df["realized_pnl_eur"].sum() if "realized_pnl_eur" in df else 0.0

    # Winrate op basis van afsluitende trades (SELL en BUY_TO_COVER)
    closes = df[df["side"].isin(["SELL","BUY_TO_COVER"])]
    wins = (closes["realized_pnl_eur"] > 0).sum() if "realized_pnl_eur" in closes else 0
    closes_n = len(closes)
    winrate = (wins / closes_n * 100.0) if closes_n else 0.0

    print("-- Samenvatting --")
    # equity uit equity.csv indien aanwezig
    if not eq.empty:
        last_row = eq.tail(1).iloc[0]
        print(f"Equity (laatst): {money(last_row.get('total_equity_eur', 0))} (op {last_row.get('date')})")
    print(f"Totaal fees:    {money(total_fees)}")
    print(f"Totaal realized: {money(total_realized)}")
    print(f"Winrate:        {winrate:.1f}%  (wins={wins} / closes={closes_n})")
    print()

    # Per pair breakdown (long & short uitgesplitst)
    print("-- Per pair --")
    for pair, g in df.groupby("pair"):
        # Long realized = som PnL van SELL
        long_real = g.loc[g["side"]=="SELL", "realized_pnl_eur"].sum()
        # Short realized = som PnL van BUY_TO_COVER
        short_real = g.loc[g["side"]=="BUY_TO_COVER", "realized_pnl_eur"].sum()
        fees = g["fee_eur"].sum()
        n_trades = len(g)
        print(f"{pair:7s} | trades={n_trades:3d} | long={money(long_real)} | short={money(short_real)} | fees={money(fees)}")

    print()
    # Laatste 10 trades (recente)
    print("-- Laatste 10 trades --")
    cols = ["timestamp","pair","side","price","amount","realized_pnl_eur"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    last10 = df.tail(10)
    for _, r in last10.iterrows():
        print(
            f"{str(r['timestamp'])[:19]} | {r['pair']:7s} | "
            f"{r['side']:12s} | €{float(r['price']):.6f} | "
            f"{float(r['amount']):.6f} | pnl={money(r['realized_pnl_eur'])}"
        )

    print(f"[runner] rapport klaar. Wacht 6/12 uur …")

if __name__ == "__main__":
    main()
