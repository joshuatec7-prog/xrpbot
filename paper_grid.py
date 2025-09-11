# paper_grid.py — Multi-coin Paper Grid (LONG + SHORT, paper) met profit-skim
# - LONG-grid (buy-low/sell-high) + SHORT-grid (sell-high/buy-lower) — alleen simulatie
# - COINS & WEIGHTS uit ENV
# - Gridband via 30d/1h P10–P90 (fallback median ± BAND_PCT)
# - Persistente state + trades.csv + equity.csv (in DATA_DIR)
# - Profit-skim houdt trading-equity rond CAPITAL_EUR; winst naar profit_pot
# - Short-exposure gelimiteerd als fractie van startkapitaal

import os, json, time, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ------------------ ENV / Defaults ------------------
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "50000"))

COINS_CSV = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
WEIGHTS_CSV = os.getenv("WEIGHTS","BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))   # 0.15% per order
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "30"))
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")
DATA_DIR    = Path(os.getenv("DATA_DIR", "/var/data"))

ORDER_SIZE_FACTOR     = float(os.getenv("ORDER_SIZE_FACTOR", "1.0"))
MIN_CASH_BUFFER_EUR   = float(os.getenv("MIN_CASH_BUFFER_EUR", "250"))
MIN_TRADE_EUR         = float(os.getenv("MIN_TRADE_EUR", "5"))
LOG_SUMMARY_SEC       = int(os.getenv("LOG_SUMMARY_SEC", "600"))

RESET_STATE = os.getenv("RESET_STATE", "false").lower() in ("1","true","yes")
WARM_START  = os.getenv("WARM_START",  "true").lower() in ("1","true","yes")

# Profit-skim
SKIM_PROFITS       = os.getenv("SKIM_PROFITS", "true").lower() in ("1","true","yes")
SKIM_INTERVAL_MIN  = int(os.getenv("SKIM_INTERVAL_MIN", "10"))
SKIM_MIN_EUR       = float(os.getenv("SKIM_MIN_EUR", "50"))

# Short-grid (paper)
ENABLE_SHORT            = os.getenv("ENABLE_SHORT", "true").lower() in ("1","true","yes")
MAX_SHORT_EXPOSURE_FRAC = float(os.getenv("MAX_SHORT_EXPOSURE_FRAC", "0.30"))  # max % van capital kort
SHORT_ORDER_FACTOR      = float(os.getenv("SHORT_ORDER_FACTOR", "1.0"))

# ------------------ Bestanden ------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE  = DATA_DIR / "state.json"
TRADES_CSV  = DATA_DIR / "trades.csv"
EQUITY_CSV  = DATA_DIR / "equity.csv"

# ------------------ Helpers ------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def append_csv(path: Path, row: List):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            if path == TRADES_CSV:
                w.writerow([
                    "timestamp","pair","side","price","amount",
                    "fee_eur","cash_eur","coin","coin_qty","realized_pnl_eur","comment"
                ])
            elif path == EQUITY_CSV:
                w.writerow(["date","total_equity_eur"])
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

def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    if weights_csv:
        for item in [x.strip() for x in weights_csv.split(",") if x.strip()]:
            if ":" in item:
                k, v = item.split(":", 1)
                try:
                    d[k.strip().upper()] = float(v)
                except Exception:
                    pass
    d = {p: d.get(p, 0.0) for p in pairs}
    s = sum(d.values())
    if s > 0:
        return {p: (d[p] / s) for p in pairs}
    eq = 1.0 / len(pairs) if pairs else 0.0
    return {p: eq for p in pairs}

# ------------------ Exchange (public) ------------------
def make_ex():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"enableRateLimit": True})
    ex.load_markets()
    return ex

# ------------------ Grid ------------------
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2: return [low, high]
    ratio = (high/low) ** (1/(n-1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    try:
        ohlcv = ex.fetch_ohlcv(pair, timeframe="1h", limit=24*30)
        if ohlcv and len(ohlcv) >= 50:
            closes = [c[4] for c in ohlcv if c and c[4] is not None]
            s = pd.Series(closes)
            p10 = float(s.quantile(0.10))
            p90 = float(s.quantile(0.90))
            if p90 > p10 > 0: return p10, p90
    except Exception:
        pass
    last = float(ex.fetch_ticker(pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def mk_grid_state(ex, pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(ex, pair)
    gl = geometric_levels(low, high, levels)
    return {
        "pair": pair,
        "low": low, "high": high, "levels": gl,
        "last_price": None,
        "long_lots": [],   # [{qty, buy_price}]
        "short_lots": [],  # [{qty, sell_price}]
        "short_notional": 0.0  # € waarde short open
    }

# ------------------ Portfolio & valuation ------------------
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    return {
        "portfolio_eur": CAPITAL_EUR,    # trading cash
        "pnl_realized": 0.0,
        "profit_eur": 0.0,               # afgeroomde winst
        "coins": {p: {"qty":0.0, "cash_alloc": CAPITAL_EUR*weights[p]} for p in pairs}
    }

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    total = state["portfolio"]["portfolio_eur"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    # short-notional is NIET bij trading_equity opgeteld; cash bevat al de short-opbrengst
    return total

# ------------------ Order sizing ------------------
def euro_per_ticket(cash_alloc: float, n_levels: int, factor: float) -> float:
    if n_levels < 2: n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(MIN_TRADE_EUR, base * factor)

# ------------------ LONG & SHORT fills (paper) ------------------
def try_fill_grid(pair: str, price_now: float, price_prev: float,
                  grid: dict, port: dict, ex) -> List[str]:
    logs = []
    levels = grid["levels"]

    # Long ticket
    long_order_eur  = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels), ORDER_SIZE_FACTOR)
    # Short ticket (kan apart geschaald worden)
    short_order_eur = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels), SHORT_ORDER_FACTOR)

    # ===== LONG: BUY op neerwaartse cross; SELL op opwaartse cross =====
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            fee_eur = long_order_eur * FEE_PCT
            if port["portfolio_eur"] - MIN_CASH_BUFFER_EUR < (long_order_eur + fee_eur):
                logs.append(f"[{pair}] BUY skip: onvoldoende EUR (buffer)."); continue
            if long_order_eur < MIN_TRADE_EUR:
                logs.append(f"[{pair}] BUY skip: onder min €{MIN_TRADE_EUR:.2f}."); continue

            qty = long_order_eur / L
            port["portfolio_eur"] -= (long_order_eur + fee_eur)
            port["coins"][pair]["qty"] += qty
            grid["long_lots"].append({"qty": qty, "buy_price": L})
            append_csv(TRADES_CSV, [now_iso(), pair,"BUY", f"{L:.6f}", f"{qty:.8f}",
                                    f"{fee_eur:.2f}", f"{port['portfolio_eur']:.2f}",
                                    pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}",
                                    f"{0.0:.2f}", "grid_buy"])
            logs.append(f"[{pair}] BUY {qty:.8f} @ €{L:.6f} | EUR_cash={port['portfolio_eur']:.2f}")

    if price_prev is not None and price_now > price_prev and grid["long_lots"]:
        crossed = [L for L in levels if price_prev < L <= price_now]
        for L in crossed:
            lot_idx = None
            for i, lot in enumerate(grid["long_lots"]):
                if L > lot["buy_price"]:
                    lot_idx = i; break
            if lot_idx is None: continue
            lot = grid["long_lots"].pop(lot_idx)
            qty = lot["qty"]
            proceeds = qty * L
            fee_eur = proceeds * FEE_PCT
            pnl = proceeds - fee_eur - (qty * lot["buy_price"])
            port["portfolio_eur"] += (proceeds - fee_eur)
            port["coins"][pair]["qty"] -= qty
            port["pnl_realized"] += pnl
            append_csv(TRADES_CSV, [now_iso(), pair,"SELL", f"{L:.6f}", f"{qty:.8f}",
                                    f"{fee_eur:.2f}", f"{port['portfolio_eur']:.2f}",
                                    pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}",
                                    f"{pnl:.2f}", "grid_sell"])
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{L:.6f} | PnL={pnl:.2f} | EUR_cash={port['portfolio_eur']:.2f}")

    # ===== SHORT (alleen paper): OPEN op OP (sell-high), CLOSE op DOWN (buy-lower) =====
    if ENABLE_SHORT:
        # OPEN short bij opwaartse cross
        if price_prev is not None and price_now > price_prev:
            crossed = [L for L in levels if price_prev < L <= price_now]
            for L in crossed:
                # exposure check
                max_short_eur = CAPITAL_EUR * MAX_SHORT_EXPOSURE_FRAC
                if grid["short_notional"] >= max_short_eur:
                    logs.append(f"[{pair}] SHORT open skip: exposure limiet."); continue
                # ticket
                ticket = min(short_order_eur, max_short_eur - grid["short_notional"])
                if ticket < MIN_TRADE_EUR:
                    continue
                qty = ticket / L
                proceeds = qty * L
                fee_open = proceeds * FEE_PCT
                # bij short-open ontvang je cash (proceeds - fee)
                port["portfolio_eur"] += (proceeds - fee_open)
                grid["short_lots"].append({"qty": qty, "sell_price": L})
                grid["short_notional"] += ticket
                append_csv(TRADES_CSV,[now_iso(),pair,"SELL", f"{L:.6f}", f"{qty:.8f}",
                                       f"{fee_open:.2f}", f"{port['portfolio_eur']:.2f}",
                                       pair.split("/")[0], f"{-qty:.8f}",
                                       f"{0.0:.2f}", "short_open"])
                logs.append(f"[{pair}] SHORT OPEN {qty:.8f} @ €{L:.6f} | cash=€{port['portfolio_eur']:.2f}")

        # CLOSE short bij neerwaartse cross (alleen winstgevende lots)
        if price_prev is not None and price_now < price_prev and grid["short_lots"]:
            crossed = [L for L in levels if price_now <= L < price_prev]
            for L in crossed:
                lot_idx = None
                for i, lot in enumerate(grid["short_lots"]):
                    if lot["sell_price"] > L:  # winstgevend: sold high, buy lower
                        lot_idx = i; break
                if lot_idx is None: continue
                lot = grid["short_lots"].pop(lot_idx)
                qty = lot["qty"]
                proceeds = qty * lot["sell_price"]   # ontvangen bij open
                fee_open = proceeds * FEE_PCT
                cost = qty * L                        # terugkopen lager
                fee_close = cost * FEE_PCT
                pnl = (proceeds - fee_open) - (cost + fee_close)
                # bij sluiten betaal je cost+fee_close
                port["portfolio_eur"] -= (cost + fee_close)
                port["pnl_realized"] += pnl
                grid["short_notional"] -= (qty * lot["sell_price"])  # benaderd ticket
                if grid["short_notional"] < 0: grid["short_notional"] = 0.0
                append_csv(TRADES_CSV,[now_iso(),pair,"BUY", f"{L:.6f}", f"{qty:.8f}",
                                       f"{fee_close:.2f}", f"{port['portfolio_eur']:.2f}",
                                       pair.split("/")[0], f"{0.0:.8f}",
                                       f"{pnl:.2f}", "short_close"])
                logs.append(f"[{pair}] SHORT CLOSE {qty:.8f} @ €{L:.6f} | PnL={pnl:.2f} | cash=€{port['portfolio_eur']:.2f}")

    grid["last_price"] = price_now
    return logs

# ------------------ Winst afromen ------------------
def skim_profits_if_needed(ex, state: dict, pairs: List[str], last_skim_ts: list):
    if not SKIM_PROFITS: return
    now = time.time()
    if now - last_skim_ts[0] < SKIM_INTERVAL_MIN * 60: return

    port = state["portfolio"]
    cash = port["portfolio_eur"]

    # holdings-waarde (longs)
    holdings_val = 0.0
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            holdings_val += qty * px

    needed_cash = max(0.0, CAPITAL_EUR - holdings_val) + MIN_CASH_BUFFER_EUR
    surplus = cash - needed_cash
    if surplus >= SKIM_MIN_EUR:
        port["portfolio_eur"] -= surplus
        port["profit_eur"]    += surplus
        print(f"[SKIM] afgeroomd €{surplus:.2f} → profit_pot; cash=€{port['portfolio_eur']:.2f} | profit=€{port['profit_eur']:.2f}")
        last_skim_ts[0] = now

# ------------------ Main ------------------
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    weights = normalize_weights(pairs, WEIGHTS_CSV)

    ex = make_ex()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs: raise SystemExit("Geen geldige markten gevonden op exchange.")

    print(f"== PAPER GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} "
          f"| fee={FEE_PCT*100:.3f}% | pairs={pairs} | factor={ORDER_SIZE_FACTOR} "
          f"| short={'ON' if ENABLE_SHORT else 'OFF'} (max {int(MAX_SHORT_EXPOSURE_FRAC*100)}%) "
          f"| buffer=€{MIN_CASH_BUFFER_EUR:.0f}")

    state = load_state() if (not RESET_STATE or WARM_START) else {}
    if "portfolio" not in state: state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state: state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    save_state(state)

    last_equity_day = None
    last_sum = 0.0
    last_skim_ts = [0.0]

    while True:
        try:
            # Dagelijkse trading-equity snapshot (excl. profit-pot)
            today = datetime.now(timezone.utc).date().isoformat()
            if today != last_equity_day:
                equity = mark_to_market(ex, state, pairs)
                append_csv(EQUITY_CSV, [today, f"{equity:.2f}"])
                last_equity_day = today

            # Per pair: prijzen ophalen en fills
            for p in pairs:
                t = ex.fetch_ticker(p)
                px = float(t["last"])
                grid = state["grids"][p]
                logs = try_fill_grid(p, px, grid["last_price"], grid, state["portfolio"], ex)
                if logs: print("\n".join(logs))

            # Profit-skim
            skim_profits_if_needed(ex, state, pairs, last_skim_ts)

            # Samenvatting
            now = time.time()
            if now - last_sum >= LOG_SUMMARY_SEC:
                trading_eq = mark_to_market(ex, state, pairs)
                cash       = state["portfolio"]["portfolio_eur"]
                realized   = state["portfolio"]["pnl_realized"]
                skimmed    = state["portfolio"]["profit_eur"]
                total_net  = trading_eq + skimmed
                total_short = sum(state["grids"][p]["short_notional"] for p in pairs)
                print(f"[SUMMARY] trading_equity=€{trading_eq:.2f} | cash=€{cash:.2f} "
                      f"| pnl_realized=€{realized:.2f} | profit_pot=€{skimmed:.2f} "
                      f"| short_exposure≈€{total_short:.2f} | total_net=€{total_net:.2f} "
                      f"| target=€{CAPITAL_EUR:.2f}")
                last_sum = now

            save_state(state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[neterr] {e}; backoff.."); time.sleep(2 + random.random())
        except Exception as e:
            print(f"[runtime] {e}"); time.sleep(5)

if __name__ == "__main__":
    main()
