# report_paper.py
# -------------------------------------------------------------
# 12-uurlijkse rapportage voor de PAPER GRID bot
# - Leest /var/data/trades.csv en /var/data/equity.csv (optioneel)
# - Maakt een tekst-rapport in /var/data/reports/
# - Robuust tegen ontbrekende kolommen/bestanden
# - Geschikt voor periodieke aanroep met --loop-hours
# -------------------------------------------------------------

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import os
import pandas as pd


# ---------- Helpers ----------
def now_local_str() -> str:
    # Lokale tijd met timezone voor bestandsnaam/logs
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def ts_for_filename() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d_%H%M")


def safe_mkdir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Als de parent (/var/data) niet bestaat of niet schrijfbaar is,
        # laten we exception bubbelen zodat je het in de logs ziet.
        raise


def load_csv(path: Path) -> pd.DataFrame:
    try:
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        # trim whitespace in kolomnamen
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        # Kapotte/lockte file? Geef lege df terug, rapport blijft draaien.
        return pd.DataFrame()


def to_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    # Converteer naar numeriek; niet-parsebare waarden worden default
    return pd.to_numeric(s, errors="coerce").fillna(default)


def coalesce_col(df: pd.DataFrame, candidates: list[str], default_value=0.0) -> pd.Series:
    """
    Zoek een kolom uit candidates; als geen van allen bestaat -> vul met default_value
    """
    for c in candidates:
        if c in df.columns:
            return to_num(df[c], default_value)
    return pd.Series([default_value] * len(df))


# ---------- Rapportage ----------
def summarize_trades(trades: pd.DataFrame) -> dict:
    """
    Verwachte kolommen (maar allemaal optioneel/robuust):
    - timestamp, pair/symbol, side, price, amount, fee_eur, realized_pnl_eur/pnl_eur
    """
    if trades.empty:
        return {
            "total_trades": 0,
            "buys": 0,
            "sells": 0,
            "fees_sum": 0.0,
            "pnl_sum": 0.0,
            "win_trades": 0,
            "loss_trades": 0,
            "avg_pnl": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "per_pair": pd.DataFrame(columns=["pair", "trades", "pnl_eur"]),
            "preview": pd.DataFrame(columns=[]),
        }

    # Normaliseer kolommen
    if "pair" in trades.columns:
        pair_col = trades["pair"].astype(str)
    elif "symbol" in trades.columns:
        pair_col = trades["symbol"].astype(str)
    else:
        pair_col = pd.Series(["UNKNOWN"] * len(trades))

    side_col = trades.get("side", pd.Series([""] * len(trades))).astype(str)

    price_col = coalesce_col(trades, ["price"])
    amount_col = coalesce_col(trades, ["amount", "qty", "quantity"])
    fee_col = coalesce_col(trades, ["fee_eur", "fee", "fees_eur"])
    pnl_col = coalesce_col(trades, ["realized_pnl_eur", "pnl_eur", "pnl"])

    # Basis stats
    total_trades = len(trades)
    buys = (side_col.str.upper() == "BUY").sum()
    sells = (side_col.str.upper() == "SELL").sum()

    fees_sum = float(fee_col.sum())
    pnl_sum = float(pnl_col.sum())

    win_trades = int((pnl_col > 0).sum())
    loss_trades = int((pnl_col < 0).sum())

    avg_pnl = float(pnl_col.mean()) if total_trades > 0 else 0.0
    best_trade = float(pnl_col.max()) if total_trades > 0 else 0.0
    worst_trade = float(pnl_col.min()) if total_trades > 0 else 0.0

    # Per-paar aggregatie
    agg = pd.DataFrame({"pair": pair_col, "pnl_eur": pnl_col})
    per_pair = (
        agg.groupby("pair", as_index=False)
        .agg(trades=("pair", "count"), pnl_eur=("pnl_eur", "sum"))
        .sort_values(["pnl_eur"], ascending=False)
    )

    # Voorbeeld: laatste 10 trades
    preview_cols = []
    for c in ["timestamp", "pair", "symbol", "side", "price", "amount", "realized_pnl_eur", "pnl_eur"]:
        if c in trades.columns:
            preview_cols.append(c)
    preview = trades[preview_cols].tail(10) if preview_cols else trades.tail(10)

    return {
        "total_trades": int(total_trades),
        "buys": int(buys),
        "sells": int(sells),
        "fees_sum": fees_sum,
        "pnl_sum": pnl_sum,
        "win_trades": win_trades,
        "loss_trades": loss_trades,
        "avg_pnl": avg_pnl,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "per_pair": per_pair,
        "preview": preview,
    }


def summarize_equity(equity: pd.DataFrame) -> dict:
    """
    Verwacht equity.csv met (minstens) kolommen: date, total_equity_eur
    """
    if equity.empty:
        return {
            "has_equity": False,
            "start": 0.0,
            "end": 0.0,
            "change": 0.0,
            "change_pct": 0.0,
            "last7_change": 0.0,
            "last7_change_pct": 0.0,
        }

    # Normaliseer
    date_col = equity.get("date", pd.Series([""] * len(equity)))
    eq_col = coalesce_col(equity, ["total_equity_eur", "equity", "equity_eur"])

    # sorteer op datum als mogelijk
    try:
        e = pd.DataFrame({"date": pd.to_datetime(date_col, errors="coerce"), "eq": eq_col})
        e = e.dropna(subset=["date"]).sort_values("date")
    except Exception:
        e = pd.DataFrame({"eq": eq_col})

    if e.empty:
        return {
            "has_equity": False,
            "start": 0.0,
            "end": 0.0,
            "change": 0.0,
            "change_pct": 0.0,
            "last7_change": 0.0,
            "last7_change_pct": 0.0,
        }

    start = float(e["eq"].iloc[0])
    end = float(e["eq"].iloc[-1])
    change = end - start
    change_pct = (change / start * 100.0) if start > 0 else 0.0

    # Laatste 7 entries (≈7 dagen bij dagelijkse snapshots)
    last7 = e.tail(7)
    lstart = float(last7["eq"].iloc[0]) if len(last7) > 0 else end
    lend = float(last7["eq"].iloc[-1]) if len(last7) > 0 else end
    last7_change = lend - lstart
    last7_change_pct = (last7_change / lstart * 100.0) if lstart > 0 else 0.0

    return {
        "has_equity": True,
        "start": start,
        "end": end,
        "change": change,
        "change_pct": change_pct,
        "last7_change": last7_change,
        "last7_change_pct": last7_change_pct,
    }


def make_report_text(tr_stats: dict, eq_stats: dict) -> str:
    lines = []
    lines.append(f"== PAPER GRID RAPPORT ==  {now_local_str()}")
    lines.append("")

    # Trades samenvatting
    lines.append("— Trades —")
    lines.append(f"Totaal trades: {tr_stats['total_trades']}")
    lines.append(f"Buys / Sells : {tr_stats['buys']} / {tr_stats['sells']}")
    lines.append(f"Totaal fees  : €{tr_stats['fees_sum']:.2f}")
    lines.append(f"Totaal PnL   : €{tr_stats['pnl_sum']:.2f}")
    lines.append(f"Win/Loss     : {tr_stats['win_trades']} / {tr_stats['loss_trades']}")
    winrate = (tr_stats["win_trades"] / tr_stats["total_trades"] * 100.0) if tr_stats["total_trades"] > 0 else 0.0
    lines.append(f"Winrate      : {winrate:.1f}%")
    lines.append(f"Gem. PnL/trd : €{tr_stats['avg_pnl']:.2f}")
    lines.append(f"Beste / slechtste trade: €{tr_stats['best_trade']:.2f} / €{tr_stats['worst_trade']:.2f}")
    lines.append("")

    # Per-paar
    lines.append("— PnL per pair —")
    per_pair: pd.DataFrame = tr_stats["per_pair"]
    if per_pair.empty:
        lines.append("(geen trades)")
    else:
        for _, r in per_pair.iterrows():
            lines.append(f"{r['pair']:<10}  trades={int(r['trades']):>4}  pnl=€{float(r['pnl_eur']):>8.2f}")
    lines.append("")

    # Equity
    lines.append("— Equity —")
    if eq_stats["has_equity"]:
        lines.append(f"Start:  €{eq_stats['start']:.2f}")
        lines.append(f"Eind:   €{eq_stats['end']:.2f}")
        lines.append(f"Δ totaal:  €{eq_stats['change']:.2f}  ({eq_stats['change_pct']:.2f}%)")
        lines.append(f"Δ laatste 7: €{eq_stats['last7_change']:.2f}  ({eq_stats['last7_change_pct']:.2f}%)")
    else:
        lines.append("(geen equity.csv gevonden)")
    lines.append("")

    # Laatste trades (preview)
    lines.append("— Laatste 10 trades (recent ->) —")
    preview: pd.DataFrame = tr_stats["preview"]
    if preview.empty:
        lines.append("(geen voorbeelden)")
    else:
        # converteer alle waarden naar string en join per rij
        prev = preview.fillna("").astype(str)
        # print header
        lines.append(" | ".join(prev.columns))
        for _, r in prev.iterrows():
            vals = [str(v) for v in r.values.tolist()]
            lines.append(" | ".join(vals))

    lines.append("")  # trailing newline
    return "\n".join(lines)


def write_report(report_dir: Path, text: str) -> Path:
    safe_mkdir(report_dir)
    out = report_dir / f"report_{ts_for_filename()}.txt"
    out.write_text(text, encoding="utf-8")
    return out


# ---------- Main ----------
def generate_once(data_dir: Path) -> Optional[Path]:
    trades_path = data_dir / "trades.csv"
    equity_path = data_dir / "equity.csv"
    report_dir = data_dir / "reports"

    trades = load_csv(trades_path)
    equity = load_csv(equity_path)

    tr_stats = summarize_trades(trades)
    eq_stats = summarize_equity(equity)

    report_text = make_report_text(tr_stats, eq_stats)
    print(report_text)

    try:
        out = write_report(report_dir, report_text)
        print(f"📄 Rapport opgeslagen: {out}")
        return out
    except Exception as e:
        print(f"⚠️  Rapport kon niet worden geschreven: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Maak periodieke rapporten voor PAPER GRID.")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("DATA_DIR", "/var/data"),
        help="Basismap met trades.csv en equity.csv (default: env DATA_DIR of /var/data)",
    )
    parser.add_argument(
        "--loop-hours",
        type=float,
        default=None,
        help="Als opgegeven: blijf draaien en maak elke N uur een rapport. Voorbeeld: --loop-hours 12",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Maak precies één rapport en stop.",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if args.once:
        generate_once(data_dir)
        return

    if args.loop_hours and args.loop_hours > 0:
        interval = max(1.0, float(args.loop_hours)) * 3600.0
        while True:
            print(f"[report] start {now_local_str()} | data={data_dir}")
            generate_once(data_dir)
            print(f"[report] klaar. Wacht {args.loop_hours} uur …")
            time.sleep(interval)
    else:
        # Default: éénmalig
        generate_once(data_dir)


if __name__ == "__main__":
    main()
