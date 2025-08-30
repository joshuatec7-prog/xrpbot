# report_paper.py
# 12-uurlijkse rapportage voor de paper grid bot
# - Leest /var/data/trades.csv en /var/data/equity.csv
# - Print overzicht naar logs
# - Schrijft rapport-bestand in /var/data/reports/
# - Optie: --loop-hours 12 blijft doorlopen

import argparse, time
from pathlib import Path
from datetime import datetime, timezone
import os
import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", "."))
TRADES_CSV = DATA_DIR / "trades.csv"
EQUITY_CSV = DATA_DIR / "equity.csv"
REPORT_DIR  = DATA_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def load_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        # leeg/niet goed – terug naar lege df
        return pd.DataFrame()

def summarize_once():
    trades = load_csv_safely(TRADES_CSV)
    equity = load_csv_safely(EQUITY_CSV)

    lines = []
    lines.append(f"===== PAPER REPORT ({ts()}) =====")

    # ----- Trades samenvatting -----
    if trades.empty:
        lines.append("Trades: geen records gevonden.")
        total_pnl = 0.0
        wins = losses = 0
    else:
        # zorg dat kolommen numeriek zijn
        for col in ["price", "amount", "pnl_eur"]:
            if col in trades.columns:
                trades[col] = pd.to_numeric(trades[col], errors="coerce")
        trades["pnl_eur"] = trades.get("pnl_eur", 0).fillna(0.0)

        total_pnl = trades["pnl_eur"].sum()
        wins   = (trades["pnl_eur"] > 0).sum()
        losses = (trades["pnl_eur"] < 0).sum()
        total  = len(trades)

        lines.append(f"Trades totaal : {total}")
        lines.append(f"Winst/verlies : +{wins} / -{losses}  (winrate: {(wins/total*100 if total else 0):.1f}%)")
        lines.append(f"Totaal gerealiseerde PnL: €{total_pnl:.2f}")

        # grootste win/verlies
        if not trades.empty:
            max_win  = trades["pnl_eur"].max()
            min_loss = trades["pnl_eur"].min()
            lines.append(f"Grootste winst : €{max_win:.2f}")
            lines.append(f"Grootste verlies: €{min_loss:.2f}")

        # PnL per pair
        if "symbol" in trades.columns:
            per_pair = trades.groupby("symbol")["pnl_eur"].sum().sort_values(ascending=False)
            lines.append("PnL per paar:")
            for sym, pnl in per_pair.items():
                lines.append(f"  - {sym}: €{pnl:.2f}")

    # ----- Equity samenvatting -----
    if equity.empty:
        lines.append("Equity: geen equity.csv gevonden (wordt 1x per dag door de bot geschreven).")
    else:
        # kolommen: date,total_equity_eur
        equity["total_equity_eur"] = pd.to_numeric(equity["total_equity_eur"], errors="coerce")
        equity = equity.dropna(subset=["total_equity_eur"])
        if len(equity) >= 1:
            start = equity["total_equity_eur"].iloc[0]
            last  = equity["total_equity_eur"].iloc[-1]
            chg_abs = last - start
            chg_pct = (chg_abs / start * 100) if start else 0
            lines.append(f"Equity start -> nu: €{start:.2f} -> €{last:.2f} (Δ €{chg_abs:.2f}, {chg_pct:.2f}%)")
        if len(equity) >= 2:
            # 12u verandering (indien er een punt 12u terug bestaat)
            lines.append("Equity historiek: laatste 5 punten:")
            tail = equity.tail(5)
            for _, row in tail.iterrows():
                lines.append(f"  - {row['date']}: €{row['total_equity_eur']:.2f}")

    # ----- Schrijf rapportbestand -----
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    rpt = REPORT_DIR / f"report_{stamp}.txt"
    rpt.write_text("\n".join(lines), encoding="utf-8")

    # print naar logs
    print("\n".join(lines))
    # schrijf ook 1 regel naar summary.csv
    sumrow = {
        "timestamp": ts(),
        "total_pnl_eur": round(total_pnl, 2),
        "wins": int(wins),
        "losses": int(losses),
    }
    sumcsv = REPORT_DIR / "summary.csv"
    write_header = not sumcsv.exists()
    pd.DataFrame([sumrow]).to_csv(sumcsv, mode="a", index=False, header=write_header)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop-hours", type=float, default=0.0,
                    help="Draai in een lus en maak elke N uur een rapport (0 = éénmalig).")
    args = ap.parse_args()

    if args.loop_hours and args.loop_hours > 0:
        interval = args.loop_hours * 3600.0
        while True:
            summarize_once()
            time.sleep(interval)
    else:
        summarize_once()

if __name__ == "__main__":
    main()
