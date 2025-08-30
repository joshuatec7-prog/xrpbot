# ------------------ Opslagpad (persistente disk) ------------------
# Gebruik /var/data als mountpad op Render. Je mag dit desgewenst via een
# ENV override nog aanpassen (DATA_DIR), maar standaard is het /var/data.
from pathlib import Path
import os, json, time, math, csv, random, statistics
from datetime import datetime, timezone
import ccxt
import pandas as pd

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Eventueel in submap bewaren zodat het netjes gescheiden is
WORK_DIR   = DATA_DIR / "paper_grid"
WORK_DIR.mkdir(exist_ok=True)

STATE_FILE = WORK_DIR / "state.json"
TRADES_CSV = WORK_DIR / "trades.csv"
EQUITY_CSV = WORK_DIR / "equity.csv"


# ====== Config via ENV ======
def _f(env: str, default: str) -> float:
    try:
        return float(os.getenv(env, default))
    except Exception:
        return float(default)

CAPITAL_EUR = _f("CAPITAL_EUR", "15000")

COINS_CSV = os.getenv(
    "COINS",
    "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
)

# Weights "PAIR:WEIGHT,PAIR:WEIGHT" — normaliseren we automatisch
WEIGHTS_CSV = os.getenv("WEIGHTS", "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10")

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = _f("BAND_PCT", "0.20")        # ±20% rondom last (fallback)
FEE_PCT     = _f("FEE_PCT", "0.0015")       # 0.15% per order
SLEEP_SEC   = _f("SLEEP_SEC", "30")
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")

# ====== Kleine helpers ======
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def append_csv(path: Path, row: List[str], header: List[str]):
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_state(state: dict):
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

# ====== Exchange (public data) ======
def make_ex():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"enableRateLimit": True})
    ex.load_markets()
    return ex

# ====== Grid bouw ======
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    r = (high / low) ** (1 / (n - 1))
    return [low * (r ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    try:
        ohlcv = ex.fetch_ohlcv(pair, timeframe="1h", limit=24*30)
        closes = [c[4] for c in ohlcv] if ohlcv else []
        if len(closes) >= 50:
            s = pd.Series(closes)
            p10 = float(s.quantile(0.10))
            p90 = float(s.quantile(0.90))
            if p90 > p10 > 0:
                return p10, p90
    except Exception:
        pass
    # fallback: ± band rond laatste prijs
    last = float(ex.fetch_ticker(pair)["last"])
    return last * (1 - BAND_PCT), last * (1 + BAND_PCT)

def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    chosen = {}
    try:
        for chunk in [x.strip() for x in weights_csv.split(",") if x.strip()]:
            k, v = chunk.split(":")
            chosen[k.strip().upper()] = float(v)
    except Exception:
        chosen = {}
    total = sum(chosen.get(p, 0.0) for p in pairs)
    if total > 0:
        return {p: chosen.get(p, 0.0) / total for p in pairs}
    eq = 1.0 / len(pairs)
    return {p: eq for p in pairs}

def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    portfolio = {
        "EUR_cash": CAPITAL_EUR,
        "coins": {p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs},
        "pnl_realized": 0.0
    }
    return portfolio

def mk_grid_state(ex, pair: str) -> dict:
    low, high = compute_band_from_history(ex, pair)
    levels = geometric_levels(low, high, GRID_LEVELS)
    return {
        "low": low, "high": high, "levels": levels,
        "last_price": None,
        "inventory_lots": []  # [{qty, buy_price}]
    }

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    total = state["portfolio"]["EUR_cash"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

# ====== Fillsimulator ======
def try_fill_grid(pair: str, price_now: float, price_prev: float, grid: dict, port: dict, ex) -> List[str]:
    logs = []
    levels = grid["levels"]

    # Ordergrootte: verdeel ~90% van alloc over (levels/2) buys
    alloc = port["coins"][pair]["cash_alloc"]
    order_eur = max(1.0, (alloc * 0.90) / max(1, len(levels)//2))

    # BUY: neerwaarts door level
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            total_cost = order_eur * (1 + FEE_PCT)
            if port["EUR_cash"] < total_cost:
                logs.append(f"[{pair}] BUY skip (te weinig EUR) cash={port['EUR_cash']:.2f}")
                continue
            qty = order_eur / L
            fee_eur = order_eur * FEE_PCT
            port["EUR_cash"] -= (order_eur + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": L})

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                 f"{port['EUR_cash']:.2f}", pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", "0.00", "grid_buy"],
                ["timestamp","pair","side","price","amount","fee_eur","eur_cash","asset","asset_qty","realized_pnl_eur","comment"]
            )
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{L:.6f} | EUR_cash={port['EUR_cash']:.2f}")

    # SELL: opwaarts door level, verkoop eerste lot met winst
    if price_prev is not None and price_now > price_prev and grid["inventory_lots"]:
        crossed = [L for L in levels if price_prev < L <= price_now]
        for L in crossed:
            idx = next((i for i, lot in enumerate(grid["inventory_lots"]) if L > lot["buy_price"]), None)
            if idx is None:
                continue
            lot = grid["inventory_lots"].pop(idx)
            qty = lot["qty"]
            proceeds = qty * L
            fee_eur = proceeds * FEE_PCT
            pnl = proceeds - fee_eur - (qty * lot["buy_price"])

            port["EUR_cash"] += (proceeds - fee_eur)
            port["coins"][pair]["qty"] -= qty
            port["pnl_realized"] += pnl

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "SELL", f"{L:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                 f"{port['EUR_cash']:.2f}", pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "grid_sell"],
                ["timestamp","pair","side","price","amount","fee_eur","eur_cash","asset","asset_qty","realized_pnl_eur","comment"]
            )
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{L:.6f} | PnL={pnl:.2f} | EUR_cash={port['EUR_cash']:.2f}")

    grid["last_price"] = price_now
    return logs

# ====== Main loop ======
def main():
    print("======================================================")
    print(" PAPER GRID BOT (paper) gestart")
    print(f"  - Data directory   : {DATA_DIR.resolve()}")
    print(f"  - State file       : {STATE_FILE.name}")
    print(f"  - Trades csv       : {TRADES_CSV.name}")
    print(f"  - Equity csv       : {EQUITY_CSV.name}")
    print(f"  - Kapitaal         : €{CAPITAL_EUR:.2f}")
    print(f"  - Grid levels      : {GRID_LEVELS}")
    print(f"  - Fee              : {FEE_PCT*100:.3f}%")
    print("======================================================")

    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    ex = make_ex()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden op exchange.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)

    state = load_state() or {}
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p)

    save_state(state)

    last_equity_day: str | None = None
    last_summary_ts = 0.0

    while True:
        try:
            # Dagelijkse equity snapshot
            today = date.today().isoformat()
            if today != last_equity_day:
                eq = mark_to_market(ex, state, pairs)
                append_csv(EQUITY_CSV, [today, f"{eq:.2f}"], ["date","total_equity_eur"])
                last_equity_day = today

            # Per pair prijzen en fills
            for p in pairs:
                px = float(ex.fetch_ticker(p)["last"])
                grid = state["grids"][p]
                logs = try_fill_grid(p, px, grid["last_price"], grid, state["portfolio"], ex)
                if logs:
                    print("\n".join(logs))

            # periodieke console-samenvatting (±60s)
            now = time.time()
            if now - last_summary_ts > 60:
                eq = mark_to_market(ex, state, pairs)
                print(f"[SUMMARY {now_iso()}] cash=€{state['portfolio']['EUR_cash']:.2f} | realized=€{state['portfolio']['pnl_realized']:.2f} | equity=€{eq:.2f}")
                last_summary_ts = now

            save_state(state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[neterr] {e}; retry...")
            time.sleep(2 + random.random())
        except Exception as e:
            print(f"[runtime] {repr(e)}")
            time.sleep(5)

if __name__ == "__main__":
    main()

