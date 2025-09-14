# live_grid.py — Multi-coin LIVE Grid (Bitvavo, spot) met:
# - Budget-cap: TARGET_CAPITAL_EUR (bv. €1.000) — winst wordt afgeroomd (niet herbelegd)
# - Bescherming bestaande holdings (PROTECT_EXISTING): we raken je start-hoeveelheden niet
# - Gridband d.m.v. 30d/1h P10–P90 (fallback median ± BAND_PCT)
# - Per-coin alloc via WEIGHTS, echte market orders, min-cost checks
# - Persistente state (live_state.json) + live_trades.csv + live_equity.csv + log

import os, json, time, csv, random, math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# --------------- ENV ---------------
API_KEY    = os.getenv("83d4d6fc710acf83996bdf182d5aed240f3fea05bc1a03bc9a620d789f3c19c6", "")
API_SECRET = os.getenv("246b15d3ac84fe5602c8538a9f4d60e81da4977fc5b0ab8c13d996245bd016beccbea128c6f0cb877edb918a8e5563d5ce47897f17b307eb2d8ec806758e91be", "")

CAPITAL_TARGET = float(os.getenv("TARGET_CAPITAL_EUR", "1000"))  # max equity in gebruik
SKIM_PROFITS   = os.getenv("SKIM_PROFITS", "true").lower() in ("1","true","yes")

COINS_CSV = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()
WEIGHTS_CSV = os.getenv("WEIGHTS", "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))
SLEEP_SEC   = float(os.getenv("SLEEP_SEC", "30"))
EXCHANGE_ID = os.getenv("EXCHANGE", "bitvavo")

ORDER_SIZE_FACTOR   = float(os.getenv("ORDER_SIZE_FACTOR", "1.8"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
MIN_TRADE_EUR       = float(os.getenv("MIN_TRADE_EUR", "5"))
LOG_SUMMARY_SEC     = int(os.getenv("LOG_SUMMARY_SEC", "21600"))  # default 6 uur

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
LOG_DIR  = Path(os.getenv("LOG_DIR", "./logs"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = DATA_DIR / "live_state.json"
TRADES_CSV = DATA_DIR / "live_trades.csv"
EQUITY_CSV = DATA_DIR / "live_equity.csv"

PROTECT_EXISTING = os.getenv("PROTECT_EXISTING", "true").lower() in ("1","true","yes")

# --------------- Helpers ---------------
def now_iso():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def append_csv(path: Path, row: List):
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            if path == TRADES_CSV:
                w.writerow(["timestamp","pair","side","price","amount","fee_eur","eur_cash","comment"])
            elif path == EQUITY_CSV:
                w.writerow(["date","equity_eur","cash_eur","skim_pot_eur"])
        w.writerow(row)

def load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default

def save_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def normalize_weights(pairs: List[str], weights_csv: str) -> Dict[str, float]:
    d: Dict[str, float] = {}
    if weights_csv:
        for item in [x.strip() for x in weights_csv.split(",") if x.strip() and ":" in x]:
            k, v = item.split(":", 1)
            try:
                d[k.strip().upper()] = float(v)
            except:
                pass
    d = {p: d.get(p, 0.0) for p in pairs}
    s = sum(d.values())
    if s > 0:
        return {p: d[p] / s for p in pairs}
    eq = 1.0/len(pairs) if pairs else 0.0
    return {p: eq for p in pairs}

# --------------- Exchange (live) ---------------
def make_ex():
    if not API_KEY or not API_SECRET:
        raise SystemExit("API_KEY/API_SECRET ontbreken.")
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({"apiKey": API_KEY, "secret": API_SECRET, "enableRateLimit": True})
    ex.load_markets()
    # Bitvavo: market buy met 'cost' param
    if hasattr(ex, "options"):
        ex.options["createMarketBuyOrderRequiresPrice"] = False
    return ex

# --------------- Grid-bouw ---------------
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2: return [low, high]
    ratio = (high/low) ** (1/(n-1))
    return [low * (ratio**i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    try:
        o = ex.fetch_ohlcv(pair, timeframe="1h", limit=24*30)
        if o and len(o) >= 50:
            closes = [c[4] for c in o if c and c[4] is not None]
            s = pd.Series(closes)
            p10 = float(s.quantile(0.10)); p90 = float(s.quantile(0.90))
            if p90 > p10 > 0:
                return p10, p90
    except Exception:
        pass
    last = float(ex.fetch_ticker(pair)["last"])
    return last*(1-BAND_PCT), last*(1+BAND_PCT)

def mk_grid_state(ex, pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(ex, pair)
    return {
        "pair": pair,
        "low": low, "high": high,
        "levels": geometric_levels(low, high, levels),
        "last_price": None,
        "lots": []  # [{qty, buy_price}]
    }

# --------------- Budget & PnL & Skim ---------------
def mark_to_market(ex, state, pairs: List[str]) -> float:
    """
    Equity = (EUR_cash_available_for_bot + SKIM_POT uitgesloten van herbeleggen?) + MV lots
    Hier: equity = EUR_cash_for_bot + MV(lots). SKIM_POT wordt NIET meegeteld als spendable.
    """
    total = state["ledger"]["eur_for_bot"]
    for p in pairs:
        qty = state["positions"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

def refresh_minima(ex, symbol):
    m = ex.markets[symbol].get("limits", {})
    min_base  = float((m.get("amount") or {}).get("min") or 0.0)
    min_quote = float((m.get("cost") or {}).get("min") or 0.0)
    return min_base, min_quote

def amount_to_precision(ex, symbol, amt):
    try:
        return float(ex.amount_to_precision(symbol, amt))
    except Exception:
        return amt

# --------------- LIVE order helpers ---------------
def market_buy_cost(ex, symbol, cost_eur: float) -> float:
    """Plaats market buy met 'cost' bij Bitvavo; return filled amount in BASE."""
    params = {"cost": float(f"{cost_eur:.2f}")}
    o = ex.create_order(symbol, "market", "buy", None, None, params)
    # fallback parsing
    filled = None
    try:
        filled = float(o.get("info", {}).get("filledAmount") or 0)
    except Exception:
        pass
    if not filled:
        # nood: schat
        last = float(ex.fetch_ticker(symbol)["last"])
        filled = cost_eur / last
    return filled

def market_sell_amount(ex, symbol, qty: float):
    qty = amount_to_precision(ex, symbol, qty)
    if qty <= 0: return 0.0
    o = ex.create_order(symbol, "market", "sell", qty)
    return qty

# --------------- Start state ---------------
def init_state(pairs: List[str], weights: Dict[str, float], ex):
    # baseline balances om bestaande holdings te beschermen
    bals = ex.fetch_balance().get("free", {})
    baseline = {"EUR": float(bals.get("EUR", 0) or 0)}
    for p in pairs:
        base, _ = p.split("/")
        baseline[base] = float(bals.get(base, 0) or 0)

    state = {
        "pairs": pairs,
        "weights": weights,
        "grids": {p: mk_grid_state(ex, p, GRID_LEVELS) for p in pairs},
        "positions": {p: {"qty": 0.0} for p in pairs},  # bot-positie, los van baseline
        "baseline": baseline,
        "ledger": {
            # geld dat de bot mag gebruiken (start: min(baseline-excess, 0) -> 0; we vullen uit live EUR tot target)
            "eur_for_bot": 0.0,          # vrij te besteden binnen budget
            "skim_pot": 0.0,             # afgeroomde winst; niet herbeleggen
        },
        "pnl_realized": 0.0,
        "last_equity_day": ""
    }
    return state

# --------------- Sizing helpers ---------------
def euro_per_ticket(alloc_eur: float, n_levels: int) -> float:
    if n_levels < 2: n_levels = 2
    base = (alloc_eur * 0.90) / (n_levels//2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

def spendable_eur(state, ex) -> float:
    """
    Bepaal hoeveel EUR we KUNNEN uitgeven zonder:
    - baseline EUR onder PROTECT_EXISTING te gebruiken (alleen 'excess' boven baseline)
    - budget (CAPITAL_TARGET) te overschrijden
    - cash buffer te negeren
    """
    bals = ex.fetch_balance().get("free", {})
    live_eur = float(bals.get("EUR", 0) or 0)
    baseline_eur = state["baseline"]["EUR"] if PROTECT_EXISTING else 0.0
    # hoeveel boven baseline beschikbaar op account:
    above_baseline = max(0.0, live_eur - baseline_eur)

    # equity van bot (alleen bot-posities + eur_for_bot)
    pairs = state["pairs"]
    eq = mark_to_market(ex, state, pairs)

    # wat is nog ruimte tot target?
    room_vs_budget = max(0.0, CAPITAL_TARGET - eq)

    # wat staat al klaar als vrije pot?
    eur_for_bot = state["ledger"]["eur_for_bot"]

    # nieuw besteedbaar: min(ruimte vs budget, 'excess' op account + bestaande eur_for_bot)
    candidate = min(room_vs_budget, above_baseline + eur_for_bot)

    # respecteer buffer
    return max(0.0, candidate - MIN_CASH_BUFFER_EUR)

# --------------- Grid loop (per tick) ---------------
def try_fill_grid_live(ex, pair, state, price_now, price_prev, min_base, min_quote):
    logs = []
    grid = state["grids"][pair]
    levels = grid["levels"]
    w = state["weights"][pair]
    alloc_eur_pair = CAPITAL_TARGET * w  # budget per pair
    order_eur = euro_per_ticket(alloc_eur_pair, len(levels))

    # BUY: neerwaartse cross (we KOPEN)
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            # check min cost
            fee_eur = order_eur * FEE_PCT
            if order_eur < max(min_quote, MIN_TRADE_EUR):
                continue
            # check spendable
            can_spend = spendable_eur(state, ex)
            if can_spend < (order_eur + fee_eur):
                continue

            # plaats order
            filled = market_buy_cost(ex, pair, order_eur)
            if filled <= 0:
                continue

            # update ledger: eur_for_bot omlaag met order+fee (als we uit pot betaalden)
            # we beschouwen alles als uit 'eur_for_bot' + excess boven baseline komt.
            # Simpel: verlaag pot met (order+fee) tot min 0.
            state["ledger"]["eur_for_bot"] = max(0.0, state["ledger"]["eur_for_bot"] - (order_eur + fee_eur))

            # positions en lots
            state["positions"][pair]["qty"] += filled
            grid["lots"].append({"qty": filled, "buy_price": L})

            append_csv(TRADES_CSV, [now_iso(), pair, "BUY", f"{L:.6f}", f"{filled:.8f}", f"{fee_eur:.2f}", "", "grid_buy"])
            logs.append(f"[{pair}] LIVE BUY {filled:.8f} @ €{L:.6f}")

    # SELL: opwaartse cross (we VERKOPEN Winst-lot) — bescherm baseline holdings
    if price_prev is not None and price_now > price_prev and grid["lots"]:
        crossed = [L for L in levels if price_prev < L <= price_now]
        for L in crossed:
            # pak eerste lot met winst
            lot_idx = None
            for i, lot in enumerate(grid["lots"]):
                if L > lot["buy_price"]:
                    lot_idx = i
                    break
            if lot_idx is None:
                continue

            lot = grid["lots"][lot_idx]
            qty = lot["qty"]

            # baseline protectie: we verkopen NOOIT onder de baseline hoeveelheid van deze coin
            base, _ = pair.split("/")
            bals = ex.fetch_balance().get("free", {})
            live_base = float(bals.get(base, 0) or 0)
            baseline_base = state["baseline"].get(base, 0.0) if PROTECT_EXISTING else 0.0

            # Max verkoopbare qty = live_base - baseline (we laten baseline staan)
            max_sellable = max(0.0, live_base - baseline_base)
            if max_sellable <= 0:
                continue

            sell_qty = min(qty, max_sellable)
            sell_qty = amount_to_precision(ex, pair, sell_qty)

            # check beurs-minima
            px = float(ex.fetch_ticker(pair)["last"])
            _, min_quote = refresh_minima(ex, pair)
            if sell_qty * px < max(min_quote, MIN_TRADE_EUR):
                continue

            sold = market_sell_amount(ex, pair, sell_qty)
            if sold <= 0:
                continue

            proceeds = sold * px
            fee_eur = proceeds * FEE_PCT
            pnl = sold * (px - lot["buy_price"]) - fee_eur

            # update positie/lot
            state["positions"][pair]["qty"] -= sold
            lot["qty"] -= sold
            if lot["qty"] <= 1e-12:
                grid["lots"].pop(lot_idx)

            # realized PnL
            if pnl > 0 and SKIM_PROFITS:
                state["ledger"]["skim_pot"] += pnl
            state["pnl_realized"] += pnl

            # vrijgekomen EUR uit verkoop: komt op je account te staan.
            # Om het budget-gedeelte bij te houden, verhogen we eur_for_bot ALLEEN
            # als equity < target (dus om terug te kunnen kopen). Winst gaat naar skim_pot.
            eq = mark_to_market(ex, state, state["pairs"])
            room = max(0.0, CAPITAL_TARGET - eq)
            # toename pot is maximaal wat nodig is om weer tot target te komen
            inc = min(proceeds - fee_eur, room)
            if inc > 0:
                state["ledger"]["eur_for_bot"] += inc

            append_csv(TRADES_CSV, [now_iso(), pair, "SELL", f"{px:.6f}", f"{sold:.8f}", f"{fee_eur:.2f}", "", f"grid_sell pnl≈€{pnl:.2f}"])
            logs.append(f"[{pair}] LIVE SELL {sold:.8f} @ €{px:.6f} | pnl≈€{pnl:.2f} | skim={state['ledger']['skim_pot']:.2f}")

    grid["last_price"] = price_now
    return logs

# --------------- Main ---------------
def main():
    print("== LIVE GRID start ==")
    ex = make_ex()
    pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige paren op exchange.")

    weights = normalize_weights(pairs, WEIGHTS_CSV)
    print(f"Pairs={pairs} | Weights={weights} | Target=€{CAPITAL_TARGET:.2f} | Skim={'ON' if SKIM_PROFITS else 'OFF'} | ProtectExisting={'ON' if PROTECT_EXISTING else 'OFF'}")

    state = load_json(STATE_FILE, None)
    if not state:
        state = init_state(pairs, weights, ex)
        # seed een eerste pot: zoveel als nodig om te starten (max CAP-target, zonder baseline aan te tasten)
        # we vullen eur_for_bot met min(excess boven baseline, CAPITAL_TARGET)
        bals = ex.fetch_balance().get("free", {})
        live_eur = float(bals.get("EUR", 0) or 0)
        baseline_eur = state["baseline"]["EUR"] if PROTECT_EXISTING else 0.0
        above_baseline = max(0.0, live_eur - baseline_eur)
        state["ledger"]["eur_for_bot"] = min(CAPITAL_TARGET, max(0.0, above_baseline - MIN_CASH_BUFFER_EUR))
        save_json(STATE_FILE, state)

    last_day = state.get("last_equity_day", "")
    last_sum = 0.0

    while True:
        try:
            # Dag-snapshot
            today = datetime.now(timezone.utc).date().isoformat()
            if today != last_day:
                eq = mark_to_market(ex, state, pairs)
                append_csv(EQUITY_CSV, [today, f"{eq:.2f}", f"{state['ledger']['eur_for_bot']:.2f}", f"{state['ledger']['skim_pot']:.2f}"])
                state["last_equity_day"] = today
                last_day = today

            # per pair
            for p in pairs:
                # minima
                min_base, min_quote = refresh_minima(ex, p)
                t = ex.fetch_ticker(p)
                px = float(t["last"])
                grid = state["grids"][p]
                logs = try_fill_grid_live(ex, p, state, px, grid["last_price"], min_base, min_quote)
                if logs:
                    print("\n".join(logs))

            # 6-uurs samenvatting
            now = time.time()
            if now - last_sum >= LOG_SUMMARY_SEC:
                eq = mark_to_market(ex, state, pairs)
                print(f"[SUMMARY] equity=€{eq:.2f} | eur_for_bot=€{state['ledger']['eur_for_bot']:.2f} | skim=€{state['ledger']['skim_pot']:.2f} | pnl_realized=€{state['pnl_realized']:.2f}")
                last_sum = now

            save_json(STATE_FILE, state)
            time.sleep(SLEEP_SEC)

        except ccxt.NetworkError as e:
            print(f"[net] {e} – backoff")
            time.sleep(2 + random.random())
        except Exception as e:
            print(f"[runtime] {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
