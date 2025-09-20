# live_grid.py — Bitvavo LIVE grid bot met strikte cap, take-profit & optionele stop-loss
#  - Geen "insufficient balance": koopt alleen als vrije EUR (incl. fee + buffer) toereikend is
#  - Respecteert exchange-minima (min quote / min base), met overrides
#  - Baseline-lock: eigen bestaande bedragen blijven met rust (LOCK_PREEXISTING_BALANCE)
#  - Neemt winst met veiligheidsmarge, toont koop/TP en verkoop-trigger/bid
#  - Optionele stop-loss om crash-risico te beperken

import csv, json, os, random, time, math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import ccxt
import pandas as pd

COL_G = "\033[92m"
COL_R = "\033[91m"
COL_C = "\033[96m"
COL_RESET = "\033[0m"


# ========== helpers ==========
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def append_csv(path: Path, row, header: Optional[List[str]] = None) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ========== exchange balance helpers ==========
def free_eur_on_exchange(ex) -> float:
    try:
        bal = ex.fetch_balance()
        return float((bal.get("free") or {}).get("EUR") or 0.0)
    except Exception:
        return 0.0


def free_base_on_exchange(ex, base: str) -> float:
    try:
        bal = ex.fetch_balance()
        return float((bal.get("free") or {}).get(base) or 0.0)
    except Exception:
        return 0.0


# ========== ENV ==========
API_KEY = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

BAND_PCT = float(os.getenv("BAND_PCT", "0.20"))
CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "1000"))
COINS_CSV = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
DATA_DIR_ARG = os.getenv("DATA_DIR", "data")
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo").strip().lower()
FEE_PCT = float(os.getenv("FEE_PCT", "0.0015"))  # 0.15%
GRID_LEVELS = int(os.getenv("GRID_LEVELS", "48"))
LOCK_PREEXISTING_BALANCE = os.getenv("LOCK_PREEXISTING_BALANCE", "true").lower() in ("1", "true", "yes")
LOG_SUMMARY_SEC = int(os.getenv("LOG_SUMMARY_SEC", "240"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
MIN_PROFIT_EUR = float(os.getenv("MIN_PROFIT_EUR", "0.10"))
MIN_PROFIT_PCT = float(os.getenv("MIN_PROFIT_PCT", "0.001"))  # 0.10%
MIN_QUOTE_EUR = float(os.getenv("MIN_QUOTE_EUR", "5"))
OPERATOR_ID = os.getenv("OPERATOR_ID", "").strip()
ORDER_SIZE_FACTOR = float(os.getenv("ORDER_SIZE_FACTOR", "1.2"))
REINVEST_PROFITS = os.getenv("REINVEST_PROFITS", "true").lower() in ("1", "true", "yes")
REPORT_EVERY_HOURS = float(os.getenv("REPORT_EVERY_HOURS", "4"))
SELL_SAFETY_PCT = float(os.getenv("SELL_SAFETY_PCT", "0.006"))  # extra marge boven netto winst-drempel
SLEEP_HEARTBEAT_SEC = int(os.getenv("SLEEP_HEARTBEAT_SEC", "300"))
SLEEP_SEC = int(os.getenv("SLEEP_SEC", "5"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0"))  # bv 0.05 = -5% hard stop
STOP_LOSS_HARD = os.getenv("STOP_LOSS_HARD", "true").lower() in ("1", "true", "yes")
WEIGHTS_CSV = os.getenv(
    "WEIGHTS",
    "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10",
).strip()

# overrides in "PAIR:MIN_BASE" bv "XRP/EUR:2.0"
def parse_overrides(s: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for it in [x.strip() for x in s.split(",") if x.strip()]:
        if ":" in it:
            k, v = it.split(":", 1)
            try:
                d[k.strip().upper()] = float(v)
            except Exception:
                pass
    return d


MIN_BASE_OVERRIDES: Dict[str, float] = {"XRP/EUR": 2.0} | parse_overrides(os.getenv("MIN_BASE_OVERRIDES", ""))

if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY / API_SECRET ontbreken.")


# ========== storage ==========
try:
    DATA_DIR = Path(DATA_DIR_ARG)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    DATA_DIR = Path("./data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = DATA_DIR / "live_state.json"
TRADES_CSV = DATA_DIR / "live_trades.csv"
EQUITY_CSV = DATA_DIR / "live_equity.csv"


# ========== exchange ==========
def make_exchange():
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    if OPERATOR_ID:
        ex.options["operatorId"] = OPERATOR_ID
    ex.load_markets()
    return ex


# ========== grid helpers ==========
def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    for it in [x.strip() for x in weights_csv.split(",") if x.strip()]:
        if ":" in it:
            k, v = it.split(":", 1)
            try:
                d[k.strip().upper()] = float(v)
            except Exception:
                pass
    s = sum(d.values()) or 0.0
    return {p: (d.get(p, 0.0) / s) if s > 0 else (1.0 / len(pairs)) for p in pairs}


def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    r = (high / low) ** (1 / (n - 1))
    return [low * (r ** i) for i in range(n)]


def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    try:
        o = ex.fetch_ohlcv(pair, timeframe="15m", limit=4 * 24 * 14)
        if o and len(o) >= 100:
            s = pd.Series([c[4] for c in o if c and c[4] is not None])
            p10 = float(s.quantile(0.10))
            p90 = float(s.quantile(0.90))
            if p90 > p10 > 0:
                return p10, p90
    except Exception:
        pass
    last = float(ex.fetch_ticker(pair)["last"])
    return last * (1 - BAND_PCT), last * (1 + BAND_PCT)


def mk_grid_state(ex, pair: str, levels: int) -> Dict:
    low, high = compute_band_from_history(ex, pair)
    return {
        "pair": pair,
        "low": low,
        "high": high,
        "levels": geometric_levels(low, high, levels),
        "last_price": None,
        "inventory_lots": [],  # list of {"qty": float, "buy_price": float}
    }


# ========== portfolio / accounting ==========
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> Dict:
    return {
        "play_cash_eur": CAPITAL_EUR,
        "cash_eur": CAPITAL_EUR,
        "pnl_realized": 0.0,
        "coins": {p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs},
    }


def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    if n_levels < 2:
        n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)  # ~ helft van levels is beneden-band
    return max(MIN_QUOTE_EUR, base * ORDER_SIZE_FACTOR)


def invested_cost_eur(state: Dict) -> float:
    tot = 0.0
    for g in state["grids"].values():
        for l in g["inventory_lots"]:
            tot += l["qty"] * l["buy_price"]
    return tot


def mark_to_market(ex, state: Dict, pairs: List[str]) -> float:
    total = state["portfolio"]["cash_eur"]
    for p in pairs:
        q = state["portfolio"]["coins"][p]["qty"]
        if q > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += q * px
    return total


def cap_now(state: Dict) -> float:
    """Huidige inzetplafond (als REINVEST_PROFITS=false blijft cap gelijk aan startkapitaal)."""
    pnl = float(state["portfolio"].get("pnl_realized", 0.0))
    return CAPITAL_EUR if REINVEST_PROFITS else max(0.0, min(CAPITAL_EUR, CAPITAL_EUR - pnl))


# ========== minima / precision ==========
def market_mins(ex, pair: str, price_now: float) -> Tuple[float, float]:
    """Returns (min_quote_cost_eur, min_base_amount)"""
    m = ex.markets.get(pair, {}) or {}
    lim = m.get("limits") or {}
    min_cost = float((lim.get("cost") or {}).get("min") or 0) or 0.0
    min_amt = float((lim.get("amount") or {}).get("min") or 0) or 0.0

    info = m.get("info") or {}
    # bitvavo specifieker:
    mc = float(info.get("minOrderInQuoteAsset") or 0) or 0.0
    ma = float(info.get("minOrderInBaseAsset") or 0) or 0.0

    min_amt = max(min_amt, ma, MIN_BASE_OVERRIDES.get(pair, 0.0))
    min_cost = max(min_cost, mc, MIN_QUOTE_EUR)
    if min_amt <= 0 and price_now > 0:
        # fallback: derive from quote-min
        min_amt = (min_cost / price_now) * 1.02
    return float(min_cost), float(min_amt)


def amount_to_precision(ex, pair: str, qty: float) -> float:
    try:
        return float(ex.amount_to_precision(pair, qty))
    except Exception:
        return qty


def best_bid_px(ex, pair: str) -> float:
    try:
        ob = ex.fetch_order_book(pair, limit=5)
        if ob and ob.get("bids"):
            return float(ob["bids"][0][0])
    except Exception:
        pass
    t = ex.fetch_ticker(pair)
    return float(t.get("bid") or t.get("last"))


# ========== P&L gates ==========
def net_gain_ok(buy_price: float, sell_avg: float, fee_pct: float, min_pct: float, min_eur: float, qty: float) -> bool:
    if buy_price <= 0 or sell_avg <= 0 or qty <= 0:
        return False
    gross = (sell_avg - buy_price) / buy_price
    net_pct = gross - 2.0 * fee_pct
    net_eur = (sell_avg - buy_price) * qty - (sell_avg * qty) * fee_pct - (buy_price * qty) * fee_pct
    return (net_pct >= min_pct) or (net_eur >= min_eur)


# ========== order wrappers ==========
def buy_market(ex, pair: str, eur_cost: float) -> Tuple[float, float, float, float]:
    """Returns (filled_qty, average_price, fee_eur, executed_eur)"""
    if eur_cost < MIN_QUOTE_EUR:
        return 0.0, 0.0, 0.0, 0.0
    params = {"cost": float(f"{eur_cost:.2f}")}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    o = ex.create_order(pair, "market", "buy", None, None, params)
    avg = float(o.get("average") or o.get("price") or 0.0)
    filled = float(o.get("filled") or o.get("info", {}).get("filledAmount") or 0.0)
    executed = avg * filled
    fee = executed * FEE_PCT
    return filled, avg, fee, executed


def sell_market(ex, pair: str, qty: float) -> Tuple[float, float, float]:
    """Returns (proceeds_eur, average_price, fee_eur)"""
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


# ========== core grid ==========
def try_grid_live(
    ex,
    pair: str,
    price_now: float,
    price_prev: Optional[float],
    state: Dict,
    grid: Dict,
    pairs: List[str],
) -> List[str]:
    logs: List[str] = []
    levels: List[float] = grid["levels"]
    if not levels:
        return logs

    port = state["portfolio"]

    # ===== BUY =====
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now < L <= price_prev]
        avail_eur = free_eur_on_exchange(ex)
        live_cap = 0.0
        # waarde van bot-inventory (excl baseline) als cap-begrenzing
        try:
            free = (ex.fetch_balance().get("free") or {})
        except Exception:
            free = {}
        baselines = state.get("baseline", {}) if LOCK_PREEXISTING_BALANCE else {}
        for p in pairs:
            base = p.split("/")[0]
            qty = max(0.0, float(free.get(base) or 0.0) - float(baselines.get(base, 0.0)))
            if qty > 0:
                px = float(ex.fetch_ticker(p)["last"])
                live_cap += qty * px

        cap = cap_now(state)
        for _ in crossed:
            ticket = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels))

            # minima
            min_quote, min_base = market_mins(ex, pair, price_now)
            required = max(min_quote, min_base * price_now)

            # start met ticket maar minimaal required
            cost = max(ticket, required)

            # rond naar boven en zet een kleine bovenmarge
            cost = math.ceil(cost * 1.01)

            # check cap & beschikbaar cash incl fee en buffer
            cost_plus_fee = cost * (1.0 + FEE_PCT)
            if invested_cost_eur(state) + cost > cap + 1e-6 or live_cap + cost > cap + 1e-6:
                logs.append(f"{COL_C}[{pair}] BUY skip: cap bereikt (cap=€{cap:.2f}).{COL_RESET}")
                continue
            if (avail_eur - MIN_CASH_BUFFER_EUR) < cost_plus_fee:
                logs.append(
                    f"{COL_C}[{pair}] BUY skip: vrije EUR te laag (free=€{avail_eur:.2f}, nodig≈€{cost_plus_fee:.2f}).{COL_RESET}"
                )
                continue

            # plaats order
            qty, avg, fee, executed = buy_market(ex, pair, cost)
            if qty <= 0 or avg <= 0:
                continue

            # sanity op minima van daadwerkelijke fill
            if qty < min_base or executed < min_quote:
                logs.append(
                    f"{COL_C}[{pair}] BUY fill < minima (amt {qty:.8f} / {min_base}, eur €{executed:.2f} / €{min_quote:.2f}); overslaan.{COL_RESET}"
                )
                continue

            # registreer lot
            grid["inventory_lots"].append({"qty": qty, "buy_price": avg})
            port["cash_eur"] -= (executed + fee)
            port["coins"][pair]["qty"] += qty
            avail_eur -= (executed + fee)
            live_cap += executed

            # take-profit target laten zien
            tp = avg * (1.0 + MIN_PROFIT_PCT + 2.0 * FEE_PCT + SELL_SAFETY_PCT)

            append_csv(
                TRADES_CSV,
                [
                    now_iso(),
                    pair,
                    "BUY",
                    f"{avg:.6f}",
                    f"{qty:.8f}",
                    f"{executed:.2f}",
                    f"{port['cash_eur']:.2f}",
                    pair.split("/")[0],
                    f"{port['coins'][pair]['qty']:.8f}",
                    "",
                    "grid_buy",
                ],
                header=[
                    "timestamp",
                    "pair",
                    "side",
                    "avg_price",
                    "qty",
                    "eur",
                    "cash_eur",
                    "base",
                    "base_qty",
                    "pnl_eur",
                    "comment",
                ],
            )
            logs.append(
                f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avg:.6f} | req≈€{cost:.2f} | fee≈€{fee:.2f} | cash=€{port['cash_eur']:.2f} | TP≈€{tp:.6f}{COL_RESET}"
            )

    # ===== SELL (TP / Stop-loss) =====
    if grid["inventory_lots"]:
        base = pair.split("/")[0]
        bot_free: Optional[float] = None
        if LOCK_PREEXISTING_BALANCE and "baseline" in state:
            bot_free = max(0.0, free_base_on_exchange(ex, base) - float(state["baseline"].get(base, 0.0)))

        min_quote, min_base = market_mins(ex, pair, price_now)
        changed = True
        while changed and grid["inventory_lots"]:
            changed = False
            # kies eerst lot dat TP haalt
            idx_tp = next(
                (
                    i
                    for i, l in enumerate(grid["inventory_lots"])
                    if net_gain_ok(l["buy_price"], price_now, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, l["qty"])
                ),
                None,
            )
            # anders check stop-loss (hard)
            idx_sl = None
            if STOP_LOSS_PCT > 0 and STOP_LOSS_HARD and idx_tp is None:
                idx_sl = next(
                    (i for i, l in enumerate(grid["inventory_lots"]) if price_now <= l["buy_price"] * (1.0 - STOP_LOSS_PCT)),
                    None,
                )

            if idx_tp is None and idx_sl is None:
                # toon even de target/bid in logs
                lot0 = grid["inventory_lots"][0]
                tp0 = lot0["buy_price"] * (1.0 + MIN_PROFIT_PCT + 2.0 * FEE_PCT + SELL_SAFETY_PCT)
                bid = best_bid_px(ex, pair)
                logs.append(f"[{pair}] SELL wait: bid €{bid:.6f} < TP €{tp0:.6f}.")
                break

            idx = idx_tp if idx_tp is not None else idx_sl
            lot = grid["inventory_lots"][idx]
            qty = lot["qty"]

            # minima
            if qty < min_base or (qty * price_now) < min_quote:
                logs.append(
                    f"[{pair}] SELL skip: lot te klein (amt {qty:.8f} / {min_base}, eur €{qty*price_now:.2f} / €{min_quote:.2f})."
                )
                break

            # baseline beschermt je eigen holdings
            if bot_free is not None and (bot_free + 1e-12) < qty:
                logs.append(f"[{pair}] SELL stop: baseline-protect ({bot_free:.8f} {base} beschikbaar).")
                break

            sell_qty = amount_to_precision(ex, pair, qty)
            if sell_qty <= 0 or sell_qty + 1e-15 < min_base:
                logs.append(f"[{pair}] SELL skip: qty {sell_qty:.8f} < min {min_base}.")
                break

            proceeds, avg, fee = sell_market(ex, pair, sell_qty)
            if proceeds > 0 and avg > 0:
                pnl = proceeds - fee - (sell_qty * lot["buy_price"])
                # als het TP-lot betreft, is pnl >= 0; bij stop-loss kan pnl < 0
                grid["inventory_lots"].pop(idx)
                port["cash_eur"] += (proceeds - fee)
                port["coins"][pair]["qty"] -= sell_qty
                port["pnl_realized"] += pnl
                if bot_free is not None:
                    bot_free -= sell_qty
                reason = "take_profit" if idx == idx_tp else "stop_loss"

                append_csv(
                    TRADES_CSV,
                    [
                        now_iso(),
                        pair,
                        "SELL",
                        f"{avg:.6f}",
                        f"{sell_qty:.8f}",
                        f"{proceeds:.2f}",
                        f"{port['cash_eur']:.2f}",
                        base,
                        f"{port['coins'][pair]['qty']:.8f}",
                        f"{pnl:.2f}",
                        reason,
                    ],
                )
                col = COL_G if pnl >= 0 else COL_R
                logs.append(
                    f"{col}[{pair}] SELL {sell_qty:.8f} @ €{avg:.6f} | proceeds=€{proceeds:.2f} | fee=€{fee:.2f} | pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}"
                )
                changed = True

    grid["last_price"] = price_now
    return logs


# ========== main loop ==========
def main():
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    ex = make_exchange()
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)
    state = load_json(STATE_FILE, {}) if STATE_FILE.exists() else {}
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    # baseline lock: noteer bestaande vrije coins zodat we ze niet “lenen”
    if LOCK_PREEXISTING_BALANCE and "baseline" not in state:
        bal = ex.fetch_balance().get("free", {})
        state["baseline"] = {p.split("/")[0]: float(bal.get(p.split("/")[0], 0) or 0.0) for p in pairs}

    save_json(STATE_FILE, state)

    print(
        f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={pct(FEE_PCT)} | "
        f"pairs={pairs} | factor={ORDER_SIZE_FACTOR} | min_profit={pct(MIN_PROFIT_PCT)} / €{MIN_PROFIT_EUR:.2f} | "
        f"sell_safety={pct(SELL_SAFETY_PCT)} | stop_loss={'uit' if STOP_LOSS_PCT<=0 else f'-{pct(STOP_LOSS_PCT)} hard={STOP_LOSS_HARD}'}"
    )

    last_report = 0.0
    last_sum = 0.0
    while True:
        try:
            eq = mark_to_market(ex, state, pairs)
            append_csv(EQUITY_CSV, [datetime.now(timezone.utc).date().isoformat(), f"{eq:.2f}"], header=["date", "total_equity_eur"])

            if time.time() - last_sum >= LOG_SUMMARY_SEC:
                since = eq - CAPITAL_EUR
                col = COL_G if since >= 0 else COL_R
                pr = state["portfolio"]["pnl_realized"]
                cash = state["portfolio"]["cash_eur"]
                inv = invested_cost_eur(state)
                fe = free_eur_on_exchange(ex)
                cap = cap_now(state)
                print(
                    f"[SUMMARY] total_eq=€{eq:.2f} | cash=€{cash:.2f} | free_EUR=€{fe:.2f} | invested_cost=€{inv:.2f} | "
                    f"cap_now=€{cap:.2f} | pnl_realized=€{pr:.2f} | since_start={col}{since:.2f}{COL_RESET}"
                )
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
                cap = cap_now(state)
                since = eq - CAPITAL_EUR
                col = COL_G if since >= 0 else COL_R
                print(
                    f"[REPORT] total_eq=€{eq:.2f} | cash=€{cash:.2f} | free_EUR=€{fe:.2f} | invested_cost=€{inv:.2f} | "
                    f"cap_now=€{cap:.2f} | pnl_realized=€{pr:.2f} | since_start={col}{since:.2f}{COL_RESET} | pairs={pairs}"
                )
                last_report = time.time()

            save_json(STATE_FILE, state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[net] {e}; backoff..")
            time.sleep(2 + random.random())
        except ccxt.BaseError as e:
            print(f"[ccxt] {e}; wacht..")
            time.sleep(5)
        except KeyboardInterrupt:
            print("Gestopt.")
            break
        except Exception as e:
            print(f"[runtime] {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
