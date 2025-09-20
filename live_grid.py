# live_grid.py — Bitvavo LIVE grid bot met harde wallet- en cap-bewakers
# - Geen BUY als vrije EUR (incl. fee + buffer) ontoereikend is → geen error 216
# - Geen BUY boven je kapitaalplafond (cap_now)
# - Verkoopt eigen gekochte lots zodra target is gehaald (met best-bid safety)
# - Logt naar data/live_trades.csv en data/live_equity.csv

import csv, json, os, time, math, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

COL_G = "\033[92m"; COL_R = "\033[91m"; COL_C = "\033[96m"; COL_RESET = "\033[0m"

def now_iso():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def pct(x: float) -> str:
    return f"{x*100:.2f}%"

def append_csv(path: Path, row, header=None):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new and header:
            w.writerow(header)
        w.writerow(row)

def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def save_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def free_eur_on_exchange(ex) -> float:
    try:
        return float((ex.fetch_balance().get("free") or {}).get("EUR") or 0.0)
    except Exception:
        return 0.0

def free_base_on_exchange(ex, base: str) -> float:
    try:
        return float((ex.fetch_balance().get("free") or {}).get(base) or 0.0)
    except Exception:
        return 0.0

# ===== ENV =====
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")
if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY / API_SECRET ontbreken.")

BAND_PCT = float(os.getenv("BAND_PCT", "0.20"))
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "1000"))
COINS_CSV = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
DATA_DIR_ARG = os.getenv("DATA_DIR", "data")
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo").strip().lower()
FEE_PCT = float(os.getenv("FEE_PCT", "0.0015"))
GRID_LEVELS = int(os.getenv("GRID_LEVELS", "32"))
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE", "true").lower() in ("1","true","yes")
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC", "240"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
MIN_PROFIT_EUR = float(os.getenv("MIN_PROFIT_EUR", "0.10"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.001"))
MIN_QUOTE_EUR = float(os.getenv("MIN_QUOTE_EUR", "5"))
OPERATOR_ID = os.getenv("OPERATOR_ID", "").strip()
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR", "1.2"))
REINVEST_PROFITS = os.getenv("REINVEST_PROFITS", "true").lower() in ("1","true","yes")
REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "4"))
SELL_SAFETY_PCT = float(os.getenv("SELL_SAFETY_PCT", "0.003"))  # 0.3%
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC", "300"))
SLEEP_SEC = int(os.getenv("SLEEP_SEC", "5"))
WEIGHTS_CSV = os.getenv("WEIGHTS", "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()

# overrides voor minimale base-amounts per markt (optioneel via ENV, bv "XRP/EUR:2.0")
def parse_overrides(s: str) -> Dict[str, float]:
    out = {}
    for it in [x.strip() for x in s.split(",") if x.strip()]:
        if ":" in it:
            k, v = it.split(":", 1)
            try:
                out[k.strip().upper()] = float(v)
            except Exception:
                pass
    return out
MIN_BASE_OVERRIDES = {"XRP/EUR": 2.0} | parse_overrides(os.getenv("MIN_BASE_OVERRIDES", ""))

# ===== storage =====
try:
    DATA_DIR = Path(DATA_DIR_ARG)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    DATA_DIR = Path("./data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = DATA_DIR / "live_state.json"
TRADES_CSV = DATA_DIR / "live_trades.csv"
EQUITY_CSV = DATA_DIR / "live_equity.csv"

# ===== exchange =====
def make_exchange():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    if OPERATOR_ID:
        ex.options["operatorId"] = OPERATOR_ID
    ex.load_markets()
    return ex

def normalize_weights(pairs, weights_csv) -> Dict[str, float]:
    d = {}
    for it in [x.strip() for x in weights_csv.split(",") if x.strip()]:
        if ":" in it:
            k, v = it.split(":", 1)
            try:
                d[k.strip().upper()] = float(v)
            except Exception:
                pass
    s = sum(d.values()) or 0.0
    return ({p: 1.0/len(pairs) for p in pairs} if s <= 0 else {p: (d.get(p, 0.0)/s) for p in pairs})

def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    r = (high/low) ** (1/(n-1))
    return [low * (r**i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    try:
        o = ex.fetch_ohlcv(pair, timeframe="15m", limit=4*24*14)  # ±14 dagen
        if o and len(o) >= 100:
            s = pd.Series([c[4] for c in o if c and c[4] is not None])
            p10 = float(s.quantile(0.10)); p90 = float(s.quantile(0.90))
            if p90 > p10 > 0:
                return p10, p90
    except Exception:
        pass
    last = float(ex.fetch_ticker(pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def mk_grid_state(ex, pair: str, levels: int) -> Dict:
    low, high = compute_band_from_history(ex, pair)
    return {
        "pair": pair,
        "low": low,
        "high": high,
        "levels": geometric_levels(low, high, levels),
        "last_price": None,
        "inventory_lots": []  # lijst van {"qty":..., "buy_price":...}
    }

def init_portfolio(pairs, weights):
    return {
        "cash_eur": CAPITAL_EUR,
        "pnl_realized": 0.0,
        "coins": {p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs}
    }

def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    if n_levels < 2:
        n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(MIN_QUOTE_EUR, base * ORDER_SIZE_FACTOR)

def mark_to_market(ex, state, pairs) -> float:
    total = state["portfolio"]["cash_eur"]
    for p in pairs:
        q = state["portfolio"]["coins"][p]["qty"]
        if q > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += q * px
    return total

def invested_cost_eur(state) -> float:
    tot = 0.0
    for g in state["grids"].values():
        for l in g["inventory_lots"]:
            tot += l["qty"] * l["buy_price"]
    return tot

def cap_now(state) -> float:
    pnl = float(state["portfolio"].get("pnl_realized", 0.0))
    return CAPITAL_EUR if REINVEST_PROFITS else max(0.0, min(CAPITAL_EUR, CAPITAL_EUR - pnl))

# Minima per markt (EUR cost en base amount)
def market_mins(ex, pair: str, price_now: float) -> Tuple[float, float]:
    m = ex.markets.get(pair, {}) or {}
    lim = m.get("limits") or {}
    min_cost = float((lim.get("cost") or {}).get("min") or 0) or 0.0
    min_amt = float((lim.get("amount") or {}).get("min") or 0) or 0.0
    info = m.get("info") or {}
    mc = float(info.get("minOrderInQuoteAsset") or 0) or 0.0
    ma = float(info.get("minOrderInBaseAsset") or 0) or 0.0
    min_amt = max(min_amt, ma, MIN_BASE_OVERRIDES.get(pair, 0.0))
    min_cost = max(min_cost, mc, MIN_QUOTE_EUR)
    if min_amt <= 0 and price_now > 0:
        min_amt = (min_cost / price_now) * 1.02
    return float(min_cost), float(min_amt)

def net_gain_ok(buy_price, sell_avg, fee_pct, min_pct, min_eur, qty) -> bool:
    if buy_price <= 0 or sell_avg <= 0 or qty <= 0:
        return False
    gross = (sell_avg - buy_price) / buy_price
    net_pct = gross - 2.0 * fee_pct
    net_eur = (sell_avg - buy_price) * qty - (sell_avg * qty) * fee_pct - (buy_price * qty) * fee_pct
    return (net_pct >= min_pct) or (net_eur >= min_eur)

def buy_market(ex, pair, eur_spend):
    if eur_spend < MIN_QUOTE_EUR:
        return 0.0, 0.0, 0.0, 0.0
    params = {"cost": float(f"{eur_spend:.2f}")}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    o = ex.create_order(pair, "market", "buy", None, None, params)
    avg = float(o.get("average") or o.get("price") or 0.0)
    filled = float(o.get("filled") or o.get("info", {}).get("filledAmount") or 0.0)
    executed = avg * filled
    fee = executed * FEE_PCT
    return filled, avg, fee, executed

def sell_market(ex, pair, qty):
    if qty <= 0:
        return 0.0, 0.0, 0.0
    params = {}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    o = ex.create_order(pair, "market", "sell", qty, None, params)
    avg = float(o.get("average") or o.get("price") or 0.0)
    proceeds = avg * qty
    fee = proceeds * FEE_PCT
    return proceeds, avg, fee

def amount_to_precision(ex, pair, qty: float) -> float:
    try:
        return float(ex.amount_to_precision(pair, qty))
    except Exception:
        return qty

def best_bid_px(ex, pair) -> float:
    try:
        ob = ex.fetch_order_book(pair, limit=5)
        if ob and ob.get("bids"):
            return float(ob["bids"][0][0])
    except Exception:
        pass
    t = ex.fetch_ticker(pair)
    return float(t.get("bid") or t.get("last"))

# ===== core =====
def bot_inventory_value_eur_from_exchange(ex, state, pairs) -> float:
    try:
        free = (ex.fetch_balance().get("free") or {})
    except Exception:
        free = {}
    baselines = state.get("baseline", {}) if LOCK_PREEXISTING_BALANCE else {}
    tot = 0.0
    for p in pairs:
        base = p.split("/")[0]
        qty = max(0.0, float(free.get(base) or 0.0) - float(baselines.get(base, 0.0) or 0.0))
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            tot += qty * px
    return tot

def try_grid_live(ex, pair, price_now, price_prev, state, grid, pairs) -> List[str]:
    logs: List[str] = []
    levels = grid["levels"]
    port = state["portfolio"]
    if not levels:
        return logs

    # BUY: alleen als prijs daalt door levels
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now < L <= price_prev]
        avail = free_eur_on_exchange(ex)
        live_cap = bot_inventory_value_eur_from_exchange(ex, state, pairs)
        cap = cap_now(state)

        for _ in crossed:
            ticket = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels))
            min_quote, min_base = market_mins(ex, pair, price_now)
            required_eur = max(min_quote, (min_base * price_now))

            # CAP-bewakers
            if invested_cost_eur(state) + required_eur > cap + 1e-9 or live_cap + required_eur > cap + 1e-9:
                logs.append(f"{COL_C}[{pair}] BUY skip: cap bereikt (cap=€{cap:.2f}).{COL_RESET}")
                continue

            # WALLET-bewaker: genoeg vrije EUR + fee + buffer
            needed = required_eur * (1.0 + FEE_PCT) + MIN_CASH_BUFFER_EUR
            if avail < needed:
                logs.append(f"{COL_C}[{pair}] BUY skip: free_EUR=€{avail:.2f} < nodig=€{needed:.2f}.{COL_RESET}")
                continue

            # Koopbedrag bepalen en afronden
            cost = min(ticket, avail - MIN_CASH_BUFFER_EUR)
            cost = max(cost, required_eur)
            cost = math.floor(cost * 100) / 100.0  # 2 decimalen

            qty, avg, fee, executed = buy_market(ex, pair, cost)
            if qty <= 0 or avg <= 0:
                continue
            if qty < min_base or executed < min_quote:
                logs.append(f"{COL_C}[{pair}] BUY fill < minima; overslaan.{COL_RESET}")
                continue

            grid["inventory_lots"].append({"qty": qty, "buy_price": avg})
            port["cash_eur"] -= (executed + fee)
            port["coins"][pair]["qty"] += qty

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "BUY", f"{avg:.6f}", f"{qty:.8f}", f"{executed:.2f}", f"{port['cash_eur']:.2f}",
                 pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
                header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"]
            )
            logs.append(f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avg:.6f} | req≈€{required_eur:.2f} | exec≈€{executed:.2f} | fee≈€{fee:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}")

    # SELL: neem winst wanneer target gehaald is
    if grid["inventory_lots"]:
        base = pair.split("/")[0]
        bot_free = None
        if LOCK_PREEXISTING_BALANCE and "baseline" in state:
            bot_free = max(0.0, free_base_on_exchange(ex, base) - float(state["baseline"].get(base, 0.0)))

        min_quote, min_base = market_mins(ex, pair, price_now)

        changed = True
        while changed and grid["inventory_lots"]:
            changed = False
            idx = next((i for i, l in enumerate(grid["inventory_lots"])
                        if net_gain_ok(l["buy_price"], price_now, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, l["qty"])),
                       None)
            if idx is None:
                break

            lot = grid["inventory_lots"][idx]
            qty = lot["qty"]

            if qty < min_base or (qty * price_now) < min_quote:
                logs.append(f"[{pair}] SELL skip: lot te klein.")
                break

            if bot_free is not None and bot_free + 1e-12 < qty:
                logs.append(f"[{pair}] SELL stop: baseline-protect.")
                break

            bid_px = best_bid_px(ex, pair)
            trigger_px = lot["buy_price"] * (1.0 + MIN_PROFIT_PCT + 2.0*FEE_PCT + SELL_SAFETY_PCT)
            if bid_px + 1e-12 < trigger_px:
                logs.append(f"[{pair}] SELL wait: bid €{bid_px:.4f} < trigger €{trigger_px:.4f}.")
                break

            sell_qty = amount_to_precision(ex, pair, qty)
            if sell_qty <= 0 or sell_qty + 1e-15 < min_base:
                logs.append(f"[{pair}] SELL skip: qty {sell_qty:.8f} < min {min_base}.")
                break

            proceeds, avg, fee = sell_market(ex, pair, sell_qty)
            if proceeds > 0 and avg > 0 and net_gain_ok(lot["buy_price"], avg, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, sell_qty):
                grid["inventory_lots"].pop(idx)
                pnl = proceeds - fee - (sell_qty * lot["buy_price"])
                port["cash_eur"] += (proceeds - fee)
                port["coins"][pair]["qty"] -= sell_qty
                port["pnl_realized"] += pnl
                if bot_free is not None:
                    bot_free -= sell_qty

                append_csv(
                    TRADES_CSV,
                    [now_iso(), pair, "SELL", f"{avg:.6f}", f"{sell_qty:.8f}", f"{proceeds:.2f}", f"{port['cash_eur']:.2f}",
                     base, f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "take_profit"]
                )
                col = COL_G if pnl >= 0 else COL_R
                logs.append(f"{col}[{pair}] SELL {sell_qty:.8f} @ €{avg:.6f} | proceeds=€{proceeds:.2f} | fee=€{fee:.2f} | pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}")
                changed = True

    grid["last_price"] = price_now
    return logs

# ===== main =====
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    ex = make_exchange()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)

    state = load_json(STATE_FILE, {})
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    if LOCK_PREEXISTING_BALANCE and "baseline" not in state:
        bal = ex.fetch_balance().get("free", {})
        state["baseline"] = {p.split("/")[0]: float(bal.get(p.split("/")[0], 0) or 0.0) for p in pairs}

    save_json(STATE_FILE, state)

    print(
        f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={pct(FEE_PCT)} | "
        f"pairs={pairs} | factor={ORDER_SIZE_FACTOR} | min_profit={pct(MIN_PROFIT_PCT)} / €{MIN_PROFIT_EUR:.2f} | "
        f"sell_safety={pct(SELL_SAFETY_PCT)}"
    )

    last_report = 0.0
    last_sum = 0.0

    while True:
        try:
            eq = mark_to_market(ex, state, pairs)
            append_csv(EQUITY_CSV, [datetime.now(timezone.utc).date().isoformat(), f"{eq:.2f}"],
                       header=["date", "total_equity_eur"])

            if time.time() - last_sum >= LOG_SUMMARY_SEC:
                since = eq - CAPITAL_EUR
                col = COL_G if since >= 0 else COL_R
                pr = state["portfolio"]["pnl_realized"]
                cash = state["portfolio"]["cash_eur"]
                inv = invested_cost_eur(state)
                fe = free_eur_on_exchange(ex)
                live = bot_inventory_value_eur_from_exchange(ex, state, pairs)
                cap = cap_now(state)
                print(f"[SUMMARY] total_eq=€{eq:.2f} | cash=€{cash:.2f} | free_EUR=€{fe:.2f} | invested_cost=€{inv:.2f} | "
                      f"live_inv=€{live:.2f} | cap_now=€{cap:.2f} | pnl_realized=€{pr:.2f} | since_start={col}{since:.2f}{COL_RESET}")
                last_sum = time.time()

            for p in pairs:
                px = float(ex.fetch_ticker(p)["last"])
                logs = try_grid_live(ex, p, px, state["grids"][p]["last_price"], state, state["grids"][p], pairs)
                if logs:
                    print("\n".join(logs))

            if time.time() - last_report >= REPORT_EVERY_HOURS * 3600:
                pr = state["portfolio"]["pnl_realized"]
                cash = state["portfolio"]["cash_eur"]
                inv = invested_cost_eur(state)
                fe = free_eur_on_exchange(ex)
                live = bot_inventory_value_eur_from_exchange(ex, state, pairs)
                cap = cap_now(state)
                since = eq - CAPITAL_EUR
                col = COL_G if since >= 0 else COL_R
                print(f"[REPORT] total_eq=€{eq:.2f} | cash=€{cash:.2f} | free_EUR=€{fe:.2f} | invested_cost=€{inv:.2f} | "
                      f"live_inv=€{live:.2f} | cap_now=€{cap:.2f} | pnl_realized=€{pr:.2f} | since_start={col}{since:.2f}{COL_RESET} | pairs={pairs}")
                last_report = time.time()

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
