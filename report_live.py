# report_live.py — LIVE rapport
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd

TRADES = Path("data/live_trades.csv")

GREEN="\033[92m"; RED="\033[91m"; BOLD="\033[1m"; RESET="\033[0m"

def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def color_amt(eur):
    eur = float(eur or 0.0)
    if eur > 0: return f"{GREEN}+€{eur:.2f}{RESET}"
    if eur < 0: return f"{RED}-€{abs(eur):.2f}{RESET}"
    return f"€{eur:.2f}"

def load_trades():
    cols = ["timestamp","pair","side","price","amount","pnl_eur"]
    if not TRADES.exists():
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(TRADES)

    if "pair" not in df.columns and "symbol" in df.columns:
        df.rename(columns={"symbol":"pair"}, inplace=True)
    if "price" not in df.columns and "avg_price" in df.columns:
        df.rename(columns={"avg_price":"price"}, inplace=True)
    if "amount" not in df.columns and "qty" in df.columns:
        df.rename(columns={"qty":"amount"}, inplace=True)
    if "pnl_eur" not in df.columns:
        df["pnl_eur"] = 0.0

    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in ("timestamp","pair","side") else 0.0

    for c in ("price","amount","pnl_eur"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    def to_date(x):
        try: return pd.to_datetime(x).date()
        except Exception: return None
    df["date"] = df["timestamp"].map(to_date)

    s = df["side"].astype(str).str.upper()
    df["is_long_close"]  = s.eq("SELL")
    df["is_short_close"] = s.eq("SHORT_CLOSE")
    df["is_close"]       = df["is_long_close"] | df["is_short_close"]

    return df[cols + ["date","is_long_close","is_short_close","is_close"]]

def main():
    print(f"{BOLD}==> LIVE RAPPORT == {ts()}{RESET}")
    df = load_trades()
    if df.empty:
        print("Nog geen trades."); return

    closes = df[df["is_close"]].copy()
    if closes.empty:
        print("Nog geen afgesloten posities."); return

    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(0, 10)]
    print("\n-- Dagelijkse winst (laatste 10 dagen) --")
    for d in sorted(days):
        pnl_dag = float(closes[closes["date"] == d]["pnl_eur"].sum() or 0.0)
        print(f"{d} | dag= {color_amt(pnl_dag)}")

    d7 = today - timedelta(days=6)
    d30 = today - timedelta(days=29)
    pnl_7  = float(closes[(closes["date"] >= d7) & (closes["date"] <= today)]["pnl_eur"].sum() or 0.0)
    pnl_30 = float(closes[(closes["date"] >= d30) & (closes["date"] <= today)]["pnl_eur"].sum() or 0.0)

    print("\n-- Periode winst --")
    print(f"7d:  {color_amt(pnl_7)}")
    print(f"30d: {color_amt(pnl_30)}")

    print("\n-- Per pair (totaal realized) --")
    by_pair = closes.groupby("pair", dropna=False)["pnl_eur"].sum().sort_values(ascending=False)
    for pair, pnl in by_pair.items():
        pair = pair if isinstance(pair, str) and pair else "?"
        print(f"{pair:<10} | pnl= {color_amt(float(pnl))}")

    if closes["is_short_close"].any():
        print("\n-- Uitsplitsing long vs short (totaal realized) --")
        long_pnl  = float(closes[closes["is_long_close"]]["pnl_eur"].sum() or 0.0)
        short_pnl = float(closes[closes["is_short_close"]]["pnl_eur"].sum() or 0.0)
        print(f"LONG : {color_amt(long_pnl)}")
        print(f"SHORT: {color_amt(short_pnl)}")

    tail = closes.tail(10).copy()
    if not tail.empty:
        print("\n-- Laatste 10 afsluitingen --")
        for _, r in tail.iterrows():
            dd  = r.get("timestamp","")
            sym = r.get("pair","?")
            side= r.get("side","")
            px  = float(r.get("price",0)  or 0)
            amt = float(r.get("amount",0) or 0)
            pnl = float(r.get("pnl_eur",0) or 0)
            print(f"{dd} | {sym:<10} | {side:<12} | €{px:.6f} | {amt:.6f} | pnl={color_amt(pnl)}")

    print("\nKlaar.\n")

if __name__ == "__main__":
    main()
