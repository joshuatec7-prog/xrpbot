# report_paper.py
# Netto resultaat = gerealiseerd (trades) + ongerealiseerd (open posities)
# Leest /var/data/{trades.csv, equity.csv, state.json}; haalt actuele prijzen op
# Optie: --loop-hours N    => draait elke N uur door (voor je runner)

import os, time, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import ccxt

# ---------------- Config / paden ----------------
DATA_DIR   = Path(os.getenv("DATA_DIR", "/var/data"))
TRADES_CSV = DATA_DIR / "trades.csv"
EQUITY_CSV = DATA_DIR / "equity.csv"   # informatief; niet nodig voor berekening
STATE_JSON = DATA_DIR / "state.json"   # bevat EUR cash + coins qty etc.

# exchange voor actuele prijzen (public)
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")

def now_local() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

# ---------------- I/O helpers ----------------
def read_trades() -> pd.DataFrame:
    if not TRADES_CSV.exists():
        return pd.DataFrame(columns=[
            "timestamp","pair","side","price","amount","fee_eur","cash_eur",
            "coin","coin_qty","realized_pnl_eur","comment"
        ])
    df = pd.read_csv(TRADES_CSV)
    # zorg dat numeriek is
    for c in ("price","amount","fee_eur","cash_eur","coin_qty","realized_pnl_eur"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    if "pair" in df.columns:
        df["pair"] = df["pair"].astype(str)
    return df

def read_state() -> dict:
    if STATE_JSON.exists():
        try:
            return json.loads(STATE_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

# ---------------- Prijsdata ----------------
def make_ex():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"enableRateLimit": True})
    ex.load_markets()
    return ex

def get_last_prices(ex, pairs: List[str]) -> Dict[str, float]:
    px = {}
    for p in pairs:
        try:
            t = ex.fetch_ticker(p)
            px[p] = float(t["last"])
        except Exception:
            px[p] = 0.0
    return px

# ---------------- Rapport-berekeningen ----------------
def compute_report() -> None:
    print(f"== PAPER GRID RAPPORT ==  {now_local()}")

    trades = read_trades()
    state  = read_state()

    # 1) GEREALISEERD
    realized = float(trades.get("realized_pnl_eur", pd.Series(dtype=float)).sum())
    fees     = float(trades.get("fee_eur", pd.Series(dtype=float)).sum())

    # 2) HUIDIGE EQUITY uit state.json + live prijzen
    # verwacht: state["portfolio"]["EUR"] en state["portfolio"]["coins"][pair]["qty"]
    capital_eur = float(os.getenv("CAPITAL_EUR", "0") or 0.0)

    eur_cash = 0.0
    coin_qty: Dict[str,float] = {}
    if state:
        try:
            eur_cash = float(state["portfolio"]["EUR"])
        except Exception:
            # oudere versie: "portfolio_eur"
            eur_cash = float(state.get("portfolio", {}).get("portfolio_eur", 0.0))
        coins = state.get("portfolio", {}).get("coins", {})
        for pair, obj in coins.items():
            try:
                coin_qty[pair] = float(obj.get("qty", 0.0))
            except Exception:
                pass

    pairs = sorted(coin_qty.keys())
    ex = make_ex() if pairs else None
    last_prices = get_last_prices(ex, pairs) if ex else {}

    # ongerealiseerd per pair = qty * last
    unrealized_value = sum( coin_qty[p] * last_prices.get(p, 0.0) for p in pairs )
    equity_now = eur_cash + unrealized_value

    # netto totaal versus startkapitaal
    net_total = equity_now - capital_eur if capital_eur else (realized - 0.0)  # fallback

    # ongerealiseerde PnL = net_total - gerealiseerd
    unrealized_pnl = net_total - realized

    # ---------------- Output ----------------
    print("\n-- Samenvatting --")
    print(f"Startkapitaal : €{capital_eur:,.2f}" if capital_eur else "Startkapitaal : (onbekend; set CAPITAL_EUR)")
    print(f"EUR cash      : €{eur_cash:,.2f}")
    if pairs:
        print(f"Coin-waarde   : €{unrealized_value:,.2f}  (mark-to-market)")
    print(f"Equity (nu)   : €{equity_now:,.2f}")
    print(f"Fees (som)    : €{fees:,.2f}")
    print(f"Gerealiseerd  : €{realized:,.2f}")
    print(f"Ongerealiseerd: €{unrealized_pnl:,.2f}")
    print(f"==> Netto resultaat (totaal): €{net_total:,.2f}")

    # Per pair (reeds gerealiseerd + huidige waarde)
    if not trades.empty or pairs:
        print("\n-- Per pair --")
        # gerealiseerd per pair uit trades
        per_pair_real = trades.groupby("pair")["realized_pnl_eur"].sum() if "pair" in trades.columns else pd.Series(dtype=float)
        per_pair_fees = trades.groupby("pair")["fee_eur"].sum() if "pair" in trades.columns else pd.Series(dtype=float)
        for p in sorted(set(per_pair_real.index).union(pairs)):
            r = float(per_pair_real.get(p, 0.0))
            f = float(per_pair_fees.get(p, 0.0))
            q = coin_qty.get(p, 0.0)
            px = last_prices.get(p, 0.0)
            value = q * px
            print(f"{p:8s} | qty={q:.6f} | px=€{px:,.4f} | value=€{value:,.2f} | realized=€{r:,.2f} | fees=€{f:,.2f}")

    # Laatste trades
    if not trades.empty:
        print("\n-- Laatste 10 trades --")
        cols = ["timestamp","pair","side","price","amount","realized_pnl_eur"]
        for _, row in trades.tail(10).iterrows():
            ts = row.get("timestamp","")
            pr = float(row.get("price",0.0))
            amt= float(row.get("amount",0.0))
            rpn= float(row.get("realized_pnl_eur",0.0))
            print(f"{ts} | {row.get('pair','?'):8s} | {str(row.get('side')):4s} | €{pr:,.4f} | {amt:.6f} | pnl=€{rpn:,.2f}")

# ---------------- Main / optional loop ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop-hours", type=float, default=0.0, help="Draai elke N uur (0 = éénmalig)")
    args = ap.parse_args()

    while True:
        try:
            compute_report()
        except Exception as e:
            print(f"[report] fout: {e}")
        if args.loop_hours and args.loop_hours > 0:
            print(f"[report] wacht {args.loop_hours} uur …")
            time.sleep(args.loop_hours * 3600)
        else:
            break

if __name__ == "__main__":
    main()
