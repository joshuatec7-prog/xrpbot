# report_live.py — LIVE rapport (spot + short aware)
# - Leest trades.csv van de live bot
# - Dagwinst in kleur (groen/rood), 7d/30d samenvatting, per pair
# - Ondersteunt zowel spot-long closes (SELL) als short-closes (SHORT_CLOSE)

from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import os

TRADES = Path("trades.csv")

# ANSI kleuren
GREEN = "\033[92m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def load_trades():
    cols = [
        "timestamp","symbol","side","price","amount",
        "reason","pnl_eur"
    ]
    if not TRADES.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(TRADES)

    # Normaliseer kolommen
    for c in ("price","amount","pnl_eur"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = 0.0

    if "symbol" not in df.columns:
        df["symbol"] = "?"

    if "side" not in df.columns:
        df["side"] = ""

    # Datum uit timestamp
    def to_date(s):
        try:
            return pd.to_datetime(s).date()
        except Exception:
            return None
    df["date"] = df["timestamp"].map(to_date)

    # Maak uniforme flags
    df["is_long_close"]  = df["side"].str.upper().eq("SELL")
    df["is_short_close"] = df["side"].str.upper().eq("SHORT_CLOSE")
    df["is_close"]       = df["is_long_close"] | df["is_short_close"]

    return df

def color_amt(eur):
    eur = float(eur or 0.0)
    if eur > 0:
        return f"{GREEN}+€{eur:.2f}{RESET}"
    if eur < 0:
        return f"{RED}-€{abs(eur):.2f}{RESET}"
    return f"€{eur:.2f}"

def main():
    print(f"{BOLD}==> LIVE RAPPORT == {ts()}{RESET}")
    df = load_trades()
    if df.empty:
        print("Nog geen trades.")
        return

    closes = df[df["is_close"]].copy()

    # Laatste 10 dagen (dag → som realized pnl)
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(0, 10)]

    print("\n-- Dagelijkse winst (laatste 10 dagen) --")
    for d in sorted(days):
        pnl_dag = float(closes[closes["date"] == d]["pnl_eur"].sum() or 0.0)
        print(f"{d} | dag= {color_amt(pnl_dag)}")

    # 7d / 30d
    d7  = today - timedelta(days=6)
    d30 = today - timedelta(days=29)
    pnl_7  = float(closes[(closes["date"] >= d7) & (closes["date"] <= today)]["pnl_eur"].sum() or 0.0)
    pnl_30 = float(closes[(closes["date"] >= d30) & (closes["date"] <= today)]["pnl_eur"].sum() or 0.0)

    print("\n-- Periode winst --")
    print(f"7d:  {color_amt(pnl_7)}")
    print(f"30d: {color_amt(pnl_30)}")

    # Per pair (totaal)
    print("\n-- Per pair (totaal realized) --")
    by_pair = closes.groupby("symbol")["pnl_eur"].sum().sort_values(ascending=False)
    for pair, pnl in by_pair.items():
        print(f"{pair:<10} | pnl= {color_amt(float(pnl))}")

    # Split long/short indien aanwezig
    has_short = closes["is_short_close"].any()
    if has_short:
        print("\n-- Uitsplitsing long vs short (totaal realized) --")
        long_pnl  = float(closes[closes["is_long_close"]]["pnl_eur"].sum() or 0.0)
        short_pnl = float(closes[closes["is_short_close"]]["pnl_eur"].sum() or 0.0)
        print(f"LONG : {color_amt(long_pnl)}")
        print(f"SHORT: {color_amt(short_pnl)}")

    # Laatste 10 closes
    tail = closes.tail(10).copy()
    if not tail.empty:
        print("\n-- Laatste 10 afsluitingen --")
        for _, r in tail.iterrows():
            dd = r.get("timestamp", "")
            sym = r.get("symbol", "?")
            side = r.get("side","")
            px = float(r.get("price",0) or 0)
            amt = float(r.get("amount",0) or 0)
            pnl = float(r.get("pnl_eur",0) or 0)
            print(f"{dd} | {sym:<10} | {side:<12} | €{px:.6f} | {amt:.6f} | pnl={color_amt(pnl)}")

    print("\nKlaar.\n")

if __name__ == "__main__":
    main()
