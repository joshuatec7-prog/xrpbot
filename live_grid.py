# live_grid.py — Bitvavo LIVE grid bot (minima hard gefixed)
# - Baseline-protect, cost-cap, live-cap
# - Minima: leest minOrderInQuoteAsset én minOrderInBaseAsset uit market.info
# - BUY: cost >= max(min_quote, min_base*price) + marge
# - SELL: slaat lots over als qty < min_base of (qty*price) < min_quote
# - Multi-sell, hele-euro cost, executed-cost logging, snelle polling

import csv, json, os, random, time, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

COL_G="\033[92m"; COL_R="\033[91m"; COL_C="\033[96m"; COL_RESET="\033[0m"

def now_iso() -> str: return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
def pct(x: float) -> str: return f"{x*100:.2f}%"

def append_csv(path: Path, row, header=None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header: w.writerow(header)
        w.writerow(row)

def load_json(path: Path, default):
    try:
        if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
    except Exception: pass
    return default

def save_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def free_eur_on_exchange(ex) -> float:
    try:
        bal = ex.fetch_balance(); free = bal.get("free") or {}
        return float(free.get("EUR") or 0.0)
    except Exception: return 0.0

def free_base_on_exchange(ex, base: str) -> float:
    try:
        bal = ex.fetch_balance(); free = bal.get("free") or {}
        return float(free.get(base) or 0.0)
    except Exception: return 0.0

def bot_inventory_value_eur_from_exchange(ex, state: dict, pairs: List[str]) -> float:
    try: free = (ex.fetch_balance().get("free") or {})
    except Exception: free = {}
    baselines = state.get("baseline", {}) if LOCK_PREEXISTING_BALANCE else {}
    tot = 0.0
    for p in pairs:
        base = p.split("/")[0]
        qty_free = float(free.get(base) or 0.0)
        baseline = float(baselines.get(base, 0.0) or 0.0)
        bot_qty = max(0.0, qty_free - baseline)
        if bot_qty > 0:
            px = float(ex.fetch_ticker(p)["last"]); tot += bot_qty * px
    return tot

# ---------- ENV ----------
API_KEY      = os.getenv("API_KEY","")
API_SECRET   = os.getenv("API_SECRET","")
BAND_PCT     = float(os.getenv("BAND_PCT","0.20"))
CAPITAL_EUR  = float(os.getenv("CAPITAL_EUR","1000"))
COINS_CSV    = os.getenv("COINS","BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
DATA_DIR_ARG = os.getenv("DATA_DIR","data")
EXCHANGE_ID  = os.getenv("EXCHANGE","bitvavo").strip().lower()
FEE_PCT      = float(os.getenv("FEE_PCT","0.0015"))
GRID_LEVELS  = int(os.getenv("GRID_LEVELS","48"))
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE","true").lower() in ("1","true","yes")
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC","240"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR","25"))
MIN_PROFIT_EUR = float(os.getenv("MIN_PROFIT_EUR","0.10"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT","0.001"))
OPERATOR_ID  = os.getenv("OPERATOR_ID","").strip()
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR","1.2"))
REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS","4"))
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC","300"))
SLEEP_SEC     = int(os.getenv("SLEEP_SEC","5"))
WEIGHTS_CSV   = os.getenv("WEIGHTS","BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()
REINVEST_PROFITS = os.getenv("REINVEST_PROFITS","true").lower() in ("1","true","yes")

def cap_now(state) -> float:
    pnl = float(state.get("portfolio",{}).get("pnl_realized",0.0))
    return CAPITAL_EUR if REINVEST_PROFITS else max(0.0, min(CAPITAL_EUR, CAPITAL_EUR - pnl))

if not API_KEY or not API_SECRET: raise SystemExit("API_KEY / API_SECRET ontbreken.")

# ---------- storage ----------
try:
    DATA_DIR = Path(DATA_DIR_ARG); DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    DATA_DIR = Path("./data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE  = DATA_DIR / "live_state.json"
TRADES_CSV  = DATA_DIR / "live_trades.csv"
EQUITY_CSV  = DATA_DIR / "live_equity.csv"

# ---------- exchange ----------
def make_exchange():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    if OPERATOR_ID: ex.options["operatorId"] = OPERATOR_ID
    ex.load_markets()
    return ex

# ---------- wegingen ----------
def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    if weights_csv:
        for item in [x.strip() for x in weights_csv.split(",") if x.strip()]:
            if ":" in item:
                k,v = item.split(":",1)
                try: d[k.strip().upper()] = float(v)
                except Exception: pass
    s = sum(d.values())
    if s <= 0: return {p: 1.0/len(pairs) for p in pairs}
    return {p: (d.get(p,0.0)/s) for p in pairs}

# ---------- grid ----------
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2: return [low, high]
    r = (high/low) ** (1/(n-1))
    return [low*(r**i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    try:
        ohlcv = ex.fetch_ohlcv(pair, timeframe="15m", limit=4*24*14)
        if ohlcv and len(ohlcv) >= 100:
            closes = [c[4] for c in ohlcv if c and c[4] is not None]
            s = pd.Series(closes); p10 = float(s.quantile(0.10)); p90 = float(s.quantile(0.90))
            if p90 > p10 > 0: return p10, p90
    except Exception: pass
    last = float(ex.fetch_ticker(pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def mk_grid_state(ex, pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(ex, pair)
    return {"pair": pair, "low": low, "high": high, "levels": geometric_levels(low, high, levels),
            "last_price": None, "inventory_lots": []}

# ---------- portfolio ----------
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    return {"play_cash_eur": CAPITAL_EUR, "cash_eur": CAPITAL_EUR, "pnl_realized": 0.0,
            "coins": {p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs}}

def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    if n_levels < 2: n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(5.0, base * ORDER_SIZE_FACTOR)

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    total = state["portfolio"]["cash_eur"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"]); total += qty * px
    return total

def invested_cost_eur(state: dict) -> float:
    tot = 0.0
    for g in state["grids"].values():
        for lot in g["inventory_lots"]: tot += lot["qty"] * lot["buy_price"]
    return tot

# ---------- minima ----------
def _to_float(x):
    try: return float(x)
    except Exception: return None

def market_mins(ex, pair: str, price_now: float | None = None) -> Tuple[float, float]:
    """Return (min_quote_eur, min_base_qty) with Bitvavo fallbacks."""
    m = ex.markets.get(pair, {}) or {}
    lim = (m.get("limits") or {})
    min_cost = _to_float((lim.get("cost") or {}).get("min"))
    min_amt  = _to_float((lim.get("amount") or {}).get("min"))
    info = (m.get("info") or {})
    mc_info = _to_float(info.get("minOrderInQuoteAsset"))
    ma_info = _to_float(info.get("minOrderInBaseAsset"))
    if not min_cost or min_cost <= 0: min_cost = mc_info if mc_info and mc_info > 0 else 5.0
    if not min_amt  or min_amt  <= 0:
        if ma_info and ma_info > 0:
            min_amt = ma_info
        else:
            # afleiding uit min_quote / prijs als laatste redmiddel
            if price_now and price_now > 0:
                min_amt = min_cost / price_now
            else:
                min_amt = 0.0
    return float(min_cost), float(min_amt)

# ---------- winst en fills ----------
def net_gain_ok(buy_price: float, sell_avg: float, fee_pct: float, min_pct: float, min_eur: float, qty: float) -> bool:
    if buy_price <= 0 or sell_avg <= 0 or qty <= 0: return False
    gross_pct = (sell_avg - buy_price) / buy_price
    net_pct   = gross_pct - 2.0 * fee_pct
    net_eur   = (sell_avg - buy_price) * qty - (sell_avg * qty) * fee_pct - (buy_price * qty) * fee_pct
    return (net_pct >= min_pct) or (net_eur >= min_eur)

def buy_market(ex, pair: str, eur: float) -> Tuple[float, float, float, float]:
    if eur < 5.0: return 0.0, 0.0, 0.0, 0.0
    params = {"cost": float(f"{eur:.2f}")}; 
    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
    order = ex.create_order(pair, "market", "buy", None, None, params)
    avgp = float(order.get("average") or order.get("price") or 0.0)
    filled = float(order.get("filled") or order.get("info", {}).get("filledAmount") or 0.0)
    executed = avgp * filled
    fee_eur = executed * FEE_PCT
    return filled, avgp, fee_eur, executed

def sell_market(ex, pair: str, qty: float) -> Tuple[float, float, float]:
    if qty <= 0: return 0.0, 0.0, 0.0
    params = {}; 
    if OPERATOR_ID: params["operatorId"] = OPERATOR_ID
    order = ex.create_order(pair, "market", "sell", qty, None, params)
    avgp = float(order.get("average") or order.get("price") or 0.0)
    proceeds = avgp * qty; fee_eur = proceeds * FEE_PCT
    return proceeds, avgp, fee_eur

# ---------- grid logica ----------
def try_grid_live(ex, pair: str, price_now: float, price_prev: float, state: dict, grid: dict, pairs_all: List[str]) -> List[str]:
    levels = grid["levels"]; port = state["portfolio"]; logs: List[str] = []
    if not levels: return logs

    # BUY
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now < L <= price_prev]
        avail_local = free_eur_on_exchange(ex)
        inv_live_cap = bot_inventory_value_eur_from_exchange(ex, state, pairs_all)
        cap = cap_now(state)
        for _ in crossed:
            ticket = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels))
            max_cost = max(0.0, avail_local - MIN_CASH_BUFFER_EUR) / (1.0 + FEE_PCT)
            cost = min(ticket, max_cost)

            min_quote, min_base = market_mins(ex, pair, price_now)
            # marge 2% + naar boven afronden
            required = max(min_quote, (min_base * price_now) * 1.02)
            cost = max(cost, required)
            cost = math.ceil(cost)   # nooit onder minimum

            if cost < max(5.0, min_quote):
                logs.append(f"{COL_C}[{pair}] BUY skip: order < minima (€≥{max(5.0,min_quote):.2f}, base≥{min_base}).{COL_RESET}")
                continue
            if invested_cost_eur(state) + cost > cap + 1e-6:
                logs.append(f"{COL_C}[{pair}] BUY skip: cost-cap (cap=€{cap:.2f}).{COL_RESET}"); continue
            if inv_live_cap + cost > cap + 1e-6:
                logs.append(f"{COL_C}[{pair}] BUY skip: live-cap (≈€{inv_live_cap:.2f}/{cap:.2f}).{COL_RESET}"); continue

            qty, avgp, fee_eur, executed = buy_market(ex, pair, cost)
            if qty <= 0 or avgp <= 0: continue

            grid["inventory_lots"].append({"qty": qty, "buy_price": avgp})
            port["cash_eur"] -= (executed + fee_eur)
            port["coins"][pair]["qty"] += qty
            avail_local -= (executed + fee_eur); inv_live_cap += executed

            append_csv(TRADES_CSV,
                [now_iso(), pair, "BUY", f"{avgp:.6f}", f"{qty:.8f}", f"{executed:.2f}", f"{port['cash_eur']:.2f}",
                 pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
                header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"])
            logs.append(f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avgp:.6f} | req=€{cost:.2f} | exec≈€{executed:.2f} | fee≈€{fee_eur:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}")

    # SELL (multi-sell + minima + baseline)
    if grid["inventory_lots"]:
        base = pair.split("/")[0]
        bot_free_avail = None
        if LOCK_PREEXISTING_BALANCE and "baseline" in state:
            free_base = free_base_on_exchange(ex, base)
            baseline = float(state["baseline"].get(base, 0.0))
            bot_free_avail = max(0.0, free_base - baseline)

        sold_any = True
        while sold_any and grid["inventory_lots"]:
            sold_any = False
            idx = next((i for i,l in enumerate(grid["inventory_lots"])
                        if net_gain_ok(l["buy_price"], price_now, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, l["qty"])), None)
            if idx is None: break
            lot = grid["inventory_lots"][idx]; qty = lot["qty"]

            min_quote, min_base = market_mins(ex, pair, price_now)
            if qty + 1e-12 < min_base:
                logs.append(f"[{pair}] SELL skip: lot {qty:.8f} < min_base {min_base}.")
                break
            if (qty * price_now) + 1e-9 < min_quote * 1.00:   # 0% marge, we zaten al boven minima bij BUY
                logs.append(f"[{pair}] SELL skip: value €{qty*price_now:.2f} < min_quote €{min_quote:.2f}.")
                break

            if bot_free_avail is not None and bot_free_avail + 1e-12 < qty:
                logs.append(f"[{pair}] SELL stop: baseline-protect ({bot_free_avail:.8f} {base} beschikbaar).")
                break

            proceeds, avgp, fee_eur = sell_market(ex, pair, qty)
            if proceeds > 0 and avgp > 0 and net_gain_ok(lot["buy_price"], avgp, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, qty):
                grid["inventory_lots"].pop(idx)
                pnl = proceeds - fee_eur - (qty * lot["buy_price"])
                port["cash_eur"] += (proceeds - fee_eur)
                port["coins"][pair]["qty"] -= qty
                port["pnl_realized"] += pnl
                if bot_free_avail is not None: bot_free_avail -= qty
                append_csv(TRADES_CSV,
                    [now_iso(), pair, "SELL", f"{avgp:.6f}", f"{qty:.8f}", f"{proceeds:.2f}", f"{port['cash_eur']:.2f}",
                     base, f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "take_profit"])
                col = COL_G if pnl >= 0 else COL_R
                logs.append(f"{col}[{pair}] SELL {qty:.8f} @ €{avgp:.6f} | proceeds=€{proceeds:.2f} | fee=€{fee_eur:.2f} | pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}")
                sold_any = True

    grid["last_price"] = price_now
    return logs

# ---------- main ----------
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    ex = make_exchange(); pairs = [p for p in pairs if p in ex.markets]
    if not pairs: raise SystemExit("Geen geldige markten gevonden.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)
    state = load_json(STATE_FILE, {}) if STATE_FILE.exists() else {}
    if "portfolio" not in state: state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state: state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]: state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    if LOCK_PREEXISTING_BALANCE and "baseline" not in state:
        bal = ex.fetch_balance().get("free", {})
        state["baseline"] = {p.split("/")[0]: float(bal.get(p.split("/")[0], 0) or 0.0) for p in pairs}

    save_json(STATE_FILE, state)

    print(f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={pct(FEE_PCT)} | "
          f"pairs={pairs} | factor={ORDER_SIZE_FACTOR} | min_profit={pct(MIN_PROFIT_PCT)} / €{MIN_PROFIT_EUR:.2f} | "
          f"reinvest_profits={REINVEST_PROFITS}")

    last_report_ts=0.0; last_summary_ts=0.0
    while True:
        try:
            eq = mark_to_market(ex, state, pairs)
            append_csv(EQUITY_CSV, [datetime.now(timezone.utc).date().isoformat(), f"{eq:.2f}"],
                       header=["date","total_equity_eur"])

            now_ts = time.time()
            if now_ts - last_summary_ts >= LOG_SUMMARY_SEC:
                since_start = eq - CAPITAL_EUR; col = COL_G if since_start >= 0 else COL_R
                pr = state["portfolio"]["pnl_realized"]; cash = state["portfolio"]["cash_eur"]
                inv_cost = invested_cost_eur(state); free_eur = free_eur_on_exchange(ex)
                live_inv = bot_inventory_value_eur_from_exchange(ex, state, pairs)
                cap = cap_now(state)
                print(f"[SUMMARY] total_eq=€{eq:.2f} | cash=€{cash:.2f} | free_EUR=€{free_eur:.2f} | "
                      f"invested_cost=€{inv_cost:.2f} | live_inv=€{live_inv:.2f} | cap_now=€{cap:.2f} | "
                      f"pnl_realized=€{pr:.2f} | since_start={col}{since_start:.2f}{COL_RESET}")
                last_summary_ts = now_ts

            for p in pairs:
                t = ex.fetch_ticker(p); px = float(t["last"])
                grid = state["grids"][p]
                logs = try_grid_live(ex, p, px, grid["last_price"], state, grid, pairs)
                if logs: print("\n".join(logs))

            if time.time() - last_report_ts >= REPORT_EVERY_HOURS*3600:
                pr = state["portfolio"]["pnl_realized"]; cash = state["portfolio"]["cash_eur"]
                inv_cost = invested_cost_eur(state); free_eur = free_eur_on_exchange(ex)
                live_inv = bot_inventory_value_eur_from_exchange(ex, state, pairs)
                cap = cap_now(state); since_start = eq - CAPITAL_EUR; col = COL_G if since_start >= 0 else COL_R
                print(f"[REPORT] total_eq=€{eq:.2f} | cash=€{cash:.2f} | free_EUR=€{free_eur:.2f} | "
                      f"invested_cost=€{inv_cost:.2f} | live_inv=€{live_inv:.2f} | cap_now=€{cap:.2f} | "
                      f"pnl_realized=€{pr:.2f} | since_start={col}{since_start:.2f}{COL_RESET} | pairs={pairs}")
                last_report_ts = time.time()

            save_json(STATE_FILE, state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[net] {e}; backoff.."); time.sleep(2 + random.random())
        except ccxt.BaseError as e:
            print(f"[ccxt] {e}; wacht.."); time.sleep(5)
        except KeyboardInterrupt:
            print("Gestopt."); break
        except Exception as e:
            print(f"[runtime] {e}"); time.sleep(5)

if __name__ == "__main__":
    main()
