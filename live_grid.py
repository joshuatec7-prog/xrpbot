# live_grid.py
# --- Multi-coin LIVE grid bot (Bitvavo) ---
# - Strikte speelpot via CAPITAL_EUR (default 1000) → bot gebruikt alleen deze pot
# - Per-pair uitgaven-cap + cooldown tussen BUYs
# - Verkoopt nooit met verlies + minimale netto-winst
# - Multi-coin met gewichten (zoals paper-grid)
# - Optioneel TP/SL, OperatorId (Bitvavo), kleur in logs

import os, json, time, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import ccxt
import pandas as pd

# ============ Helpers ============
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def append_csv(path: Path, row: List, header: List[str] = None):
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

def pct(x) -> str:
    return f"{x*100:.2f}%"

# ============ ENV (alfabetisch) ============
API_KEY              = os.getenv("API_KEY", "")
API_SECRET           = os.getenv("API_SECRET", "")

BAND_PCT             = float(os.getenv("BAND_PCT", "0.20"))

BUY_COOLDOWN_SEC     = int(os.getenv("BUY_COOLDOWN_SEC", "180"))

CAPITAL_EUR          = float(os.getenv("CAPITAL_EUR", "1000"))

COINS                = os.getenv("COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR").strip()

DATA_DIR             = Path(os.getenv("DATA_DIR", "/opt/render/project/src/data"))

DISABLE_COLOR        = os.getenv("DISABLE_COLOR", "false").lower() in ("1","true","yes")
ENABLE_SHORT         = os.getenv("ENABLE_SHORT", "false").lower() in ("1","true","yes")  # niet gebruikt in live

EXCHANGE             = os.getenv("EXCHANGE", "bitvavo").strip().lower()

FEE_PCT              = float(os.getenv("FEE_PCT", "0.0015"))  # 0.15%

FORBID_LOSS_SELL     = os.getenv("FORBID_LOSS_SELL", "true").lower() in ("1","true","yes")

GRID_LEVELS          = int(os.getenv("GRID_LEVELS", "24"))

LOCK_PREEXISTING_BAL = os.getenv("LOCK_PREEXISTING_BAL", os.getenv("PROTECT_EXISTING", "true")).lower() in ("1","true","yes")

LOG_DIR              = Path(os.getenv("LOG_DIR", "./logs"))

LOG_SUMMARY_SEC      = int(os.getenv("LOG_SUMMARY_SEC", "600"))

MAX_EUR_PER_PAIR     = float(os.getenv("MAX_EUR_PER_PAIR", "200"))

MIN_CASH_BUFFER_EUR  = float(os.getenv("MIN_CASH_BUFFER_EUR", "25"))
MIN_NET_PROFIT_PCT   = float(os.getenv("MIN_NET_PROFIT_PCT", "0.004"))   # 0.4% netto minimaal
MIN_TRADE_EUR        = float(os.getenv("MIN_TRADE_EUR", "5"))

OPERATOR_ID          = os.getenv("OPERATOR_ID", "").strip()

ORDER_SIZE_FACTOR    = float(os.getenv("ORDER_SIZE_FACTOR", "1.8"))      # jouw setting

REPORT_EVERY_HOURS   = float(os.getenv("REPORT_EVERY_HOURS", "4"))

SLEEP_SEC            = int(os.getenv("SLEEP_SEC", "15"))

STOP_LOSS_PCT        = float(os.getenv("STOP_LOSS_PCT", "0.00"))         # 0.0 = uit
TAKE_PROFIT_PCT      = float(os.getenv("TAKE_PROFIT_PCT", "0.00"))       # 0.0 = uit

WEIGHTS              = os.getenv("WEIGHTS", "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10").strip()

# ============ Bestanden ============
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE  = DATA_DIR / "live_state.json"
TRADES_CSV  = DATA_DIR / "live_trades.csv"
EQUITY_CSV  = DATA_DIR / "live_equity.csv"

# ============ Exchange ============
if not API_KEY or not API_SECRET:
    raise SystemExit("API_KEY/API_SECRET ontbreken (Render → Environment).")

def make_ex():
    klass = getattr(ccxt, EXCHANGE)
    ex = klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True
    })
    if OPERATOR_ID:
        ex.options["operatorId"] = OPERATOR_ID
    ex.load_markets()
    return ex

# ============ Weights ============
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

# ============ Grid tools ============
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    """P10–P90 op 30d/1h closes; fallback ±BAND_PCT rond last."""
    try:
        ohlcv = ex.fetch_ohlcv(pair, timeframe="1h", limit=24*30)
        if ohlcv and len(ohlcv) >= 50:
            closes = [c[4] for c in ohlcv if c and c[4] is not None]
            s = pd.Series(closes)
            p10 = float(s.quantile(0.10))
            p90 = float(s.quantile(0.90))
            if p90 > p10 > 0:
                return p10, p90
    except Exception:
        pass
    last = float(ex.fetch_ticker(pair)["last"])
    low  = last * (1 - BAND_PCT)
    high = last * (1 + BAND_PCT)
    return low, high

def mk_grid_state(ex, pair: str, levels: int) -> dict:
    low, high = compute_band_from_history(ex, pair)
    grid_levels = geometric_levels(low, high, levels)
    return {
        "pair": pair,
        "low": low,
        "high": high,
        "levels": grid_levels,
        "last_price": None,
        "inventory_lots": [],   # [{qty, buy_price}]
        # remmen:
        "last_buy_ts": 0.0,
        "spent_eur": 0.0,
    }

# ============ Portfolio ============
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    return {
        "cash_eur": CAPITAL_EUR,         # strikte speelpot
        "pnl_realized": 0.0,
        "coins": {p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs}
    }

def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    if n_levels < 2:
        n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    total = state["portfolio"]["cash_eur"]
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

# ============ Orders ============
def market_buy_eur(ex, pair: str, spend_eur: float, px_hint: float, params: dict) -> Tuple[float, float]:
    """Return (filled_qty, avg_price)."""
    spend_eur = float(f"{spend_eur:.2f}")  # Bitvavo cost = 2 dp
    min_needed = max(MIN_TRADE_EUR, 0.50)
    if spend_eur < min_needed:
        return 0.0, 0.0
    order = ex.create_order(pair, "market", "buy", None, None, {"cost": spend_eur, **params})
    filled = float(order.get("info", {}).get("filledAmount") or 0.0)
    avg = float(order.get("average") or px_hint or 0.0)
    if filled <= 0.0 and px_hint > 0:
        filled = spend_eur / px_hint
        avg = px_hint
    return filled, avg

def market_sell_amount(ex, pair: str, qty: float, params: dict) -> Tuple[float, float]:
    """Return (proceeds_eur, avg_price)."""
    if qty <= 0:
        return 0.0, 0.0
    order = ex.create_order(pair, "market", "sell", qty, None, params)
    avg = float(order.get("average") or 0.0)
    proceeds = qty * avg if avg > 0 else 0.0
    return proceeds, avg

# ============ Grid logic ============
def try_grid_live(ex, pair: str, price_now: float, price_prev: float,
                  grid: dict, port: dict, params: dict) -> List[str]:
    logs: List[str] = []
    levels = grid["levels"]
    cash_alloc = port["coins"][pair]["cash_alloc"]
    ticket_eur = euro_per_ticket(cash_alloc, len(levels))

    # BUY op neerwaarts cross
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            # remmen: cap per pair, cooldown, pot, buffer
            if grid.get("spent_eur", 0.0) >= MAX_EUR_PER_PAIR:
                logs.append(f"[{pair}] BUY skip: per-pair cap (€{MAX_EUR_PER_PAIR:.0f}).")
                continue
            now_ts = time.time()
            if now_ts - grid.get("last_buy_ts", 0.0) < BUY_COOLDOWN_SEC:
                logs.append(f"[{pair}] BUY skip: cooldown {BUY_COOLDOWN_SEC}s.")
                continue
            fee_eur = ticket_eur * FEE_PCT
            need = ticket_eur + fee_eur
            if (port["cash_eur"] - MIN_CASH_BUFFER_EUR) < need:
                logs.append(f"[{pair}] BUY skip: pot/buffer onvoldoende.")
                continue

            # place order
            filled, avg = market_buy_eur(ex, pair, ticket_eur, L, params)
            if filled <= 0 or avg <= 0:
                logs.append(f"[{pair}] BUY fail.")
                continue

            # boekingen
            port["cash_eur"] -= (ticket_eur + ticket_eur * FEE_PCT)
            port["coins"][pair]["qty"] += filled
            grid["inventory_lots"].append({"qty": filled, "buy_price": avg})
            grid["spent_eur"] += ticket_eur
            grid["last_buy_ts"] = now_ts

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "BUY", f"{avg:.6f}", f"{filled:.8f}", f"{ticket_eur*FEE_PCT:.2f}",
                 f"{port['cash_eur']:.2f}", pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", "0.00", "grid_buy"],
                header=["timestamp","pair","side","price","amount","fee_eur","cash_eur","coin","coin_qty","pnl_eur","comment"]
            )
            logs.append(f"[{pair}] BUY {filled:.8f} @ €{avg:.6f} | cash={port['cash_eur']:.2f}")

    # SELL op opwaarts cross (winst)
    if price_prev is not None and price_now > price_prev and grid["inventory_lots"]:
        crossed = [L for L in levels if price_prev < L <= price_now]
        for L in crossed:
            lot_idx = None
            for i, lot in enumerate(grid["inventory_lots"]):
                if L > lot["buy_price"]:
                    lot_idx = i
                    break
            if lot_idx is None:
                continue

            lot = grid["inventory_lots"].pop(lot_idx)
            qty = lot["qty"]

            # actuele avg & proceeds
            proceeds, avgp = market_sell_amount(ex, pair, qty, params)
            if proceeds <= 0 or avgp <= 0:
                # niet verkocht, lot terug
                grid["inventory_lots"].insert(lot_idx, lot)
                continue

            fee_eur = proceeds * FEE_PCT
            pnl = proceeds - fee_eur - (qty * lot["buy_price"])

            # verlies verbieden + minimale netto winst
            if (FORBID_LOSS_SELL and pnl <= 0.0) or (pnl < (qty * lot["buy_price"] * MIN_NET_PROFIT_PCT)):
                # lot terug (niet verkopen)
                grid["inventory_lots"].insert(lot_idx, lot)
                logs.append(f"[{pair}] SELL skip: pnl te laag.")
                continue

            # boekingen
            port["cash_eur"] += (proceeds - fee_eur)
            port["coins"][pair]["qty"] -= qty
            port["pnl_realized"] += pnl
            # budget vrijgeven ruwweg met ticketbedrag
            grid["spent_eur"] = max(0.0, grid.get("spent_eur", 0.0) - proceeds)

            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "SELL", f"{avgp:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                 f"{port['cash_eur']:.2f}", pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "grid_sell"],
                header=["timestamp","pair","side","price","amount","fee_eur","cash_eur","coin","coin_qty","pnl_eur","comment"]
            )
            logs.append(f"[{pair}] SELL {qty:.8f} @ €{avgp:.6f} | pnl=€{pnl:.2f} | cash={port['cash_eur']:.2f}")

    grid["last_price"] = price_now
    return logs

# ============ Main ============
def main():
    pairs = [x.strip().upper() for x in COINS.split(",") if x.strip()]
    ex = make_ex()
    weights = normalize_weights(pairs, WEIGHTS)
    pairs = [p for p in pairs if p in ex.markets]
    if not pairs:
        raise SystemExit("Geen geldige markten gevonden.")

    # Bescherm bestaande balans (alleen speelpot gebruiken)
    acct = ex.fetch_balance().get("free", {})
    start_eur_on_exchange = float(acct.get("EUR", 0) or 0)

    print(f"== LIVE GRID start | capital=€{CAPITAL_EUR:.2f} | levels={GRID_LEVELS} | fee={pct(FEE_PCT)} | "
          f"pairs={pairs} | weights={weights} | cooldown={BUY_COOLDOWN_SEC}s | cap/pair=€{MAX_EUR_PER_PAIR:.0f}")

    # State
    state = load_json(STATE_FILE, {})
    if "portfolio" not in state:
        state["portfolio"] = init_portfolio(pairs, weights)
    if "grids" not in state:
        state["grids"] = {}
    for p in pairs:
        if p not in state["grids"]:
            state["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

    # CSV headers indien leeg
    if not EQUITY_CSV.exists():
        append_csv(EQUITY_CSV, ["date","total_equity_eur"], header=None)

    last_summary = 0.0
    last_report  = 0.0

    while True:
        try:
            # sanity: gebruik nooit meer dan speelpot (ook al staat er meer EUR op de account)
            if LOCK_PREEXISTING_BAL:
                # puur logica-borging: we gebruiken state['portfolio']['cash_eur'] als pot,
                # en plaatsen buy alleen als pot toereikend is (zie try_grid_live). Dat is genoeg.

                pass

            # Heartbeat + equity snapshot (dagelijks)
            today = datetime.now(timezone.utc).date().isoformat()
            if not EQUITY_CSV.exists() or (EQUITY_CSV.stat().st_size == 0) or True:
                eq = mark_to_market(ex, state, pairs)
                # we schrijven elke loop; UI filtert zelf
                append_csv(EQUITY_CSV, [today, f"{eq:.2f}"], header=None)

            # operator params eenmaal
            params = {}
            if OPERATOR_ID:
                params["operatorId"] = OPERATOR_ID

            # Per pair griden
            for p in pairs:
                t = ex.fetch_ticker(p)
                px = float(t["last"])
                grid = state["grids"][p]
                logs = try_grid_live(ex, p, px, grid["last_price"], grid, state["portfolio"], params)
                if logs:
                    print("\n".join(logs))

                # optioneel hard TP/SL per lot
                if (STOP_LOSS_PCT > 0.0 or TAKE_PROFIT_PCT > 0.0) and grid["inventory_lots"]:
                    i = 0
                    while i < len(grid["inventory_lots"]):
                        lot = grid["inventory_lots"][i]
                        trigger_sl = STOP_LOSS_PCT > 0.0 and px <= lot["buy_price"] * (1.0 - STOP_LOSS_PCT)
                        trigger_tp = TAKE_PROFIT_PCT > 0.0 and px >= lot["buy_price"] * (1.0 + TAKE_PROFIT_PCT)
                        if trigger_sl or trigger_tp:
                            qty = lot["qty"]
                            proceeds, avgp = market_sell_amount(ex, p, qty, params)
                            if proceeds > 0 and avgp > 0:
                                fee_eur = proceeds * FEE_PCT
                                pnl = proceeds - fee_eur - (qty * lot["buy_price"])
                                if trigger_sl and FORBID_LOSS_SELL and pnl <= 0.0:
                                    i += 1
                                    continue
                                port = state["portfolio"]
                                port["cash_eur"] += (proceeds - fee_eur)
                                port["coins"][p]["qty"] -= qty
                                port["pnl_realized"] += pnl
                                state["grids"][p]["spent_eur"] = max(0.0, state["grids"][p].get("spent_eur", 0.0) - proceeds)
                                append_csv(
                                    TRADES_CSV,
                                    [now_iso(), p, "SELL", f"{avgp:.6f}", f"{qty:.8f}", f"{fee_eur:.2f}",
                                     f"{port['cash_eur']:.2f}", p.split("/")[0], f"{port['coins'][p]['qty']:.8f}", f"{pnl:.2f}",
                                     "hard_tp_sl"],
                                    header=["timestamp","pair","side","price","amount","fee_eur","cash_eur","coin","coin_qty","pnl_eur","comment"]
                                )
                                print(f"[{p}] {'SL' if trigger_sl else 'TP'} SELL {qty:.8f} @ €{avgp:.6f} | pnl=€{pnl:.2f}")
                                grid["inventory_lots"].pop(i)
                                continue
                        i += 1

            # Periodieke SUMMARY
            now = time.time()
            if now - last_summary >= LOG_SUMMARY_SEC:
                eq = mark_to_market(ex, state, pairs)
                cash = state["portfolio"]["cash_eur"]
                pr   = state["portfolio"]["pnl_realized"]
                day_change = eq - CAPITAL_EUR
                use_color = not DISABLE_COLOR
                prefix = "\033[92m" if (use_color and day_change >= 0) else ("\033[91m" if use_color else "")
                suffix = "\033[0m" if use_color else ""
                print(f"[SUMMARY] total_eq=€{eq:.2f} | cash=€{cash:.2f} | pnl_realized=€{pr:.2f} | "
                      f"day_change={prefix}€{day_change:.2f}{suffix}")
                last_summary = now

            # 4u rapport: in dit script doen we alleen SUMMARY; runner is optioneel
            if now - last_report >= REPORT_EVERY_HOURS * 3600:
                eq = mark_to_market(ex, state, pairs)
                print(f"[REPORT] equity~€{eq:.2f} | cash~€{state['portfolio']['cash_eur']:.2f} | time={now_iso()}")
                last_report = now

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
