# report_paper.py — uitgebreid dagrapport (gerealiseerd + schatting ongerealiseerd)
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, sys, json
import pandas as pd

import ccxt  # voor actuele prijs t.b.v. ongerealiseerd

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
TRADES   = DATA_DIR / "trades.csv"
EQUITY   = DATA_DIR / "equity.csv"
STATE    = DATA_DIR / "state.json"

EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")
COINS_CSV   = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()

def ts():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def load_exchange():
    try:
        klass = getattr(ccxt, EXCHANGE_ID)
        ex = klass({"enableRateLimit": True})
        ex.load_markets()
        return ex
    except Exception:
        return None

def last_close_price(ex, pair: str) -> float:
    if ex is None:
        return float('nan')
    try:
        t = ex.fetch_ticker(pair)
        return float(t["last"])
    except Exception:
        return float('nan')

def unrealized_from_state(ex) -> float:
    """Schat ongerealiseerde PnL voor LONG lots uit state.json."""
    if not STATE.exists():
        return 0.0
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return 0.0

    grids = (state or {}).get("grids", {})
    if not grids:
        return 0.0

    total_unreal = 0.0
    for pair, g in grids.items():
        lots = g.get("inventory_lots", []) or []
        if not lots:
            continue
        px = last_close_price(ex, pair)
        if not px or px != px:
            continue  # NaN of 0
        for lot in lots:
            try:
                qty = float(lot["qty"])
                buy = float(lot["buy_price"])
                total_unreal += qty * (px - buy)  # fees buiten beschouwing
            except Exception:
                pass
    return total_unreal

def main():
    print(f"==> PAPIER GRID RAPPORT ++  {ts()}")

    # ===== Gerealiseerde winst per dag =====
    if not TRADES.exists():
        print("(geen trades)")
        return

    df = pd.read_csv(TRADES)
    # schoonmaken
    for col in ("price","amount","fee_eur","realized_pnl_eur"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # timestamp -> date
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.date
    else:
        df["date"] = pd.NaT

    realized_day = df.groupby("date")["realized_pnl_eur"].sum().sort_index()

    print("\n-- Dagelijkse gerealiseerde winst (laatste 10 dagen) --")
    tail = realized_day.tail(10)
    for d, v in tail.items():
        print(f"{d} | realized=€{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    # periode sommen
    today = datetime.now().date()
    d7   = today - timedelta(days=6)
    d30  = today - timedelta(days=29)

    r7  = realized_day.loc[realized_day.index >= d7].sum() if len(realized_day) else 0.0
    r30 = realized_day.loc[realized_day.index >= d30].sum() if len(realized_day) else 0.0

    print("\n-- Periode winst --")
    print(f"7d:  €{r7:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    print(f"30d: €{r30:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

    # ===== Laatste equity =====
    latest_eq = "-"
    if EQUITY.exists():
        try:
            e = pd.read_csv(EQUITY)
            if "total_equity_eur" in e.columns and len(e):
                latest_eq = f"€{float(e['total_equity_eur'].iloc[-1]):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            pass

    # ===== Ongerealiseerde schatting uit state =====
    ex = load_exchange()
    unreal = unrealized_from_state(ex)
    unreal_txt = f"€{unreal:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    # print samenvatting
    print("\n-- Samenvatting --")
    print(f"Equity (laatst): {latest_eq}")
    print(f"Ongerealiseerd (schatting longs): {unreal_txt}")

    # Optioneel: target tonen
    target = os.getenv("TARGET_EQUITY_EUR", "50000")
    try:
        tgt = float(target)
        print(f"Target trading equity: €{tgt:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    except Exception:
        pass

if __name__ == "__main__":
    main()
