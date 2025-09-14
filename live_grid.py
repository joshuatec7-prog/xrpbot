# live_grid.py — Multi-coin LIVE grid bot voor Bitvavo
# - Budgetcap (CAPITAL_EUR) + lock pre-existing balances
# - Skim: winst direct afgeroomd (niet herinvesteren)
# - Meerdere pairs, gewichten, grid via 30d/1h P10–P90 (fallback median±BAND_PCT)
# - Kleur-logs (ANSI)
# - Rapport iedere REPORT_EVERY_HOURS (default 4 uur)
# - CSV: /var/data/{trades.csv,equity.csv}, state in live_state.json

import os, time, json, csv, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import ccxt
import pandas as pd

# ------------------ ENV / Defaults ------------------
API_KEY     = os.getenv("API_KEY", "").strip()
API_SECRET  = os.getenv("API_SECRET", "").strip()
OPERATOR_ID = os.getenv("OPERATOR_ID", "").strip()

CAPITAL_EUR = float(os.getenv("CAPITAL_EUR", "1000"))

COINS_CSV = os.getenv(
    "COINS", "BTC/EUR,ETH/EUR,SOL/EUR,XRP/EUR,LTC/EUR"
).strip()

WEIGHTS_CSV = os.getenv(
    "WEIGHTS", "BTC/EUR:0.35,ETH/EUR:0.30,SOL/EUR:0.15,XRP/EUR:0.10,LTC/EUR:0.10"
).strip()

GRID_LEVELS = int(os.getenv("GRID_LEVELS", "24"))
BAND_PCT    = float(os.getenv("BAND_PCT", "0.20"))
FEE_PCT     = float(os.getenv("FEE_PCT", "0.0015"))   # 0.15%
MIN_TRADE_EUR       = float(os.getenv("MIN_TRADE_EUR", "5"))
MIN_CASH_BUFFER_EUR = float(os.getenv("MIN_CASH_BUFFER_EUR", "50"))
ORDER_SIZE_FACTOR   = float(os.getenv("ORDER_SIZE_FACTOR", "1.0"))
SLEEP_SEC           = float(os.getenv("SLEEP_SEC", "15"))
REPORT_EVERY_HOURS  = float(os.getenv("REPORT_EVERY_HOURS", "4"))

LOCK_PREEXISTING_BAL = os.getenv("LOCK_PREEXISTING_BAL", "true").lower() in ("1","true","yes")

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))

# kleur
DISABLE_COLOR = os.getenv("DISABLE_COLOR", "false").lower() in ("1","true","yes")
class C:
    RESET  = "" if DISABLE_COLOR else "\033[0m"
    BOLD   = "" if DISABLE_COLOR else "\033[1m"
    RED    = "" if DISABLE_COLOR else "\033[91m"
    GREEN  = "" if DISABLE_COLOR else "\033[92m"
    YELLOW = "" if DISABLE_COLOR else "\033[93m"
    CYAN   = "" if DISABLE_COLOR else "\033[96m"

def posneg_color(v: float) -> str:
    return C.GREEN if v > 0 else (C.RED if v < 0 else "")

def fmt_pnl(prefix: str, v: float) -> str:
    col = posneg_color(v)
    return f"{prefix}{col}€{v:.2f}{C.RESET}"

# ------------------ Bestanden ------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = DATA_DIR / "live_state.json"
TRADES_CSV = DATA_DIR / "trades.csv"
EQUITY_CSV = DATA_DIR / "equity.csv"

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

# ------------------ Exchange ------------------
def make_ex():
    if not API_KEY or not API_SECRET:
        raise SystemExit("API_KEY of API_SECRET ontbreekt.")
    ex = ccxt.bitvavo({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
    })
    # sommige accounts vereisen operatorId verplicht:
    if OPERATOR_ID:
        ex.options = ex.options or {}
        ex.options["operatorId"] = OPERATOR_ID
    try:
        ex.load_markets()
    except Exception as e:
        raise SystemExit(f"Kon markets niet laden: {e}")
    # sanity check: balance opvragen -> faalt als key niet actief/geen leesrecht
    try:
        _ = ex.fetch_balance()
    except ccxt.BaseError as e:
        raise SystemExit(f"API check faalt: {e}")
    return ex

# ------------------ Grid bouw ------------------
def geometric_levels(low: float, high: float, n: int) -> List[float]:
    if low <= 0 or high <= 0 or n < 2:
        return [low, high]
    ratio = (high / low) ** (1 / (n - 1))
    return [low * (ratio ** i) for i in range(n)]

def compute_band_from_history(ex, pair: str) -> Tuple[float, float]:
    """P10–P90 van 30d 1h closes; fallback median ± BAND_PCT."""
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
    }

# ------------------ Portfolio & budget ------------------
def init_portfolio(pairs: List[str], weights: Dict[str, float]) -> dict:
    return {
        "budget_eur": CAPITAL_EUR,       # maximaal inzetbaar budget (wordt NIET boven capital gezet door skim)
        "pnl_realized": 0.0,             # cumulatieve gerealiseerde winst
        "skimmed_total": 0.0,            # hoeveel afgeroomd
        "coins": { p: {"qty": 0.0, "cash_alloc": CAPITAL_EUR * weights[p]} for p in pairs },
        "start_wallet_eur": 0.0,         # eur die je al had voor start (lock pre-existing)
        "start_wallet_coins": {},        # per coin qty vóór start (lock pre-existing)
    }

def euro_per_ticket(cash_alloc: float, n_levels: int) -> float:
    """Ticketgrootte gebaseerd op allocatie en grid levels."""
    if n_levels < 2:
        n_levels = 2
    base = (cash_alloc * 0.90) / (n_levels // 2)
    return max(MIN_TRADE_EUR, base * ORDER_SIZE_FACTOR)

def avail_eur_for_bot(ex, state, start_wallet_eur: float, budget_eur: float) -> float:
    """Vrij EUR dat de bot mag gebruiken (beschermt je bestaande wallet EUR)."""
    bal = ex.fetch_balance().get("free", {})
    eur = float(bal.get("EUR", 0) or 0)
    if LOCK_PREEXISTING_BAL:
        eur_bot = max(0.0, eur - start_wallet_eur)    # alleen boven je start-EUR gebruiken
    else:
        eur_bot = eur
    # cap door budget
    return max(0.0, min(eur_bot, budget_eur))

def mark_to_market(ex, state: dict, pairs: List[str]) -> float:
    """Equity = budget cash (niet direct te zien) + waarde holdings + externe wallet EUR (locked)."""
    # We tonen "equity" als: EUR vrij in wallet + waarde bot-holdings
    bal = ex.fetch_balance().get("free", {})
    eur_free = float(bal.get("EUR", 0) or 0)
    total = eur_free
    for p in pairs:
        qty = state["portfolio"]["coins"][p]["qty"]
        if qty > 0:
            px = float(ex.fetch_ticker(p)["last"])
            total += qty * px
    return total

# ------------------ Order helpers ------------------
def market_buy_cost(ex, pair: str, euro_cost: float) -> float:
    """Market BUY via cost param; returns filled base amount."""
    euro_cost = float(f"{euro_cost:.2f}")  # Bitvavo cost met 2 decimals
    params = {"cost": euro_cost}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    o = ex.create_order(pair, "market", "buy", None, None, params)
    filled = None
    try:
        filled = float(o.get("info", {}).get("filledAmount"))
    except Exception:
        pass
    if not filled:
        # fallback schatting
        px = float(ex.fetch_ticker(pair)["last"])
        filled = euro_cost / px
    return filled

def market_sell_amount(ex, pair: str, qty: float) -> float:
    """Market SELL met amount; returns sold qty."""
    params = {}
    if OPERATOR_ID:
        params["operatorId"] = OPERATOR_ID
    ex.create_order(pair, "market", "sell", qty, None, params)
    return qty

# ------------------ Skim (winst afromen) ------------------
def maybe_do_skim(ex, st: dict, pairs: List[str]):
    """Als equity > start_equity + budget → room het verschil af en verlaag budget (niet herinvesteren)."""
    # Simpele implementatie: als EUR vrij + holdings waarde > (start_wallet_eur + budget), verhoog skim en houd budget gelijk.
    eq = mark_to_market(ex, st, pairs)
    target = st["portfolio"]["start_wallet_eur"] + st["portfolio"]["budget_eur"]
    overshoot = eq - target
    if overshoot > 1.0:
        st["portfolio"]["skimmed_total"] += overshoot
        # budget blijft hetzelfde (we herinvesteren niet boven budget)
        print(f"{C.BOLD}{C.YELLOW}[SKIM]{C.RESET} afgeroomd €{overshoot:.2f} | "
              f"totaal_skim={C.YELLOW}€{st['portfolio']['skimmed_total']:.2f}{C.RESET}")

# ------------------ Fillsimulator -> REAL orders ------------------
def try_fill_grid(ex, pair: str, price_now: float, price_prev: Optional[float],
                  grid: dict, port: dict, start_wallet_eur: float) -> List[str]:
    logs = []
    levels = grid["levels"]
    cash_alloc = port["coins"][pair]["cash_alloc"]
    order_eur  = euro_per_ticket(cash_alloc, len(levels))

    # BUY: neerwaartse cross (koop 1 lot per level)
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now <= L < price_prev]
        for L in crossed:
            # genoeg EUR (buffer + budgetcap + lock-preexisting)
            free_eur = avail_eur_for_bot(ex, st, start_wallet_eur, port["budget_eur"])
            fee_eur  = order_eur * FEE_PCT
            if free_eur - MIN_CASH_BUFFER_EUR < (order_eur + fee_eur):
                logs.append(f"[{pair}] BUY skip: onvoldoende EUR (buffer/budget).")
                continue
            if order_eur < MIN_TRADE_EUR:
                logs.append(f"[{pair}] BUY skip: < min trade €{MIN_TRADE_EUR:.2f}.")
                continue
            # plaats LIVE koop
            qty = market_buy_cost(ex, pair, order_eur)
            port["coins"][pair]["qty"] += qty
            grid["inventory_lots"].append({"qty": qty, "buy_price": L})
            logs.append(f"{C.CYAN}[{pair}] BUY{C.RESET} {qty:.8f} @ €{L:.6f}")

            append_csv(TRADES_CSV, [
                now_iso(), pair, "BUY", f"{L:.6f}", f"{qty:.8f}",
                f"{(order_eur*FEE_PCT):.2f}", "-", pair.split("/")[0],
                f"{port['coins'][pair]['qty']:.8f}", f"{0.0:.2f}", "grid_buy"
            ])

    # SELL: opwaartse cross (verkoop 1 winnende lot)
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
            proceeds = qty * L
            fee_eur = proceeds * FEE_PCT
            pnl = proceeds - fee_eur - (qty * lot["buy_price"])

            # LIVE verkoop
            market_sell_amount(ex, pair, qty)
            port["coins"][pair]["qty"] -= qty
            port["pnl_realized"] += pnl

            logs.append(f"[{pair}] SELL {qty:.8f} @ €{L:.6f} | " + fmt_pnl("pnl=", pnl))

            append_csv(TRADES_CSV, [
                now_iso(), pair, "SELL", f"{L:.6f}", f"{qty:.8f}",
                f"{fee_eur:.2f}", "-", pair.split("/")[0],
                f"{port['coins'][pair]['qty']:.8f}",
                f"{pnl:.2f}", "grid_sell"
            ])

    grid["last_price"] = price_now
    return logs

# ------------------ Main ------------------
print(
  f"{C.BOLD}== LIVE GRID start{C.RESET} | capital=€{CAPITAL_EUR:.2f} | "
  f"levels={GRID_LEVELS} | fee={FEE_PCT*100:.3f}% | buffer=€{MIN_CASH_BUFFER_EUR:.0f}"
)

ex = make_ex()

pairs = [x.strip().upper() for x in COINS_CSV.split(",") if x.strip()]
pairs = [p for p in pairs if p in ex.markets]
if not pairs:
    raise SystemExit("Geen geldige markten gevonden.")

weights = normalize_weights(pairs, WEIGHTS_CSV)

# state laden/aanmaken
st = load_state()
if "portfolio" not in st:
    st["portfolio"] = init_portfolio(pairs, weights)

# initialiseer start_wallet-limieten (lock preexisting)
if st["portfolio"]["start_wallet_eur"] == 0.0:
    bal = ex.fetch_balance().get("free", {})
    st["portfolio"]["start_wallet_eur"] = float(bal.get("EUR", 0) or 0)
    # coins die je al had
    start_coins = {}
    for p in pairs:
        base = p.split("/")[0]
        start_coins[base] = float(bal.get(base, 0) or 0)
    st["portfolio"]["start_wallet_coins"] = start_coins

# grids
if "grids" not in st:
    st["grids"] = {}
for p in pairs:
    if p not in st["grids"]:
        st["grids"][p] = mk_grid_state(ex, p, GRID_LEVELS)

save_state(st)

last_equity_day = None
last_report_ts  = 0.0
last_sum        = 0.0

# Eerste summary
eq0 = mark_to_market(ex, st, pairs)
print(f"[SUMMARY] equity=€{eq0:.2f} | cash=€{avail_eur_for_bot(ex, st, st['portfolio']['start_wallet_eur'], st['portfolio']['budget_eur']):.2f} | "
      f"{fmt_pnl('pnl_realized=', st['portfolio']['pnl_realized'])} | "
      f"skim={C.YELLOW}€{st['portfolio']['skimmed_total']:.2f}{C.RESET}")

while True:
    try:
        # Dagelijkse equity snapshot
        today = datetime.now(timezone.utc).date().isoformat()
        if today != last_equity_day:
            equity = mark_to_market(ex, st, pairs)
            append_csv(EQUITY_CSV, [today, f"{equity:.2f}"])
            last_equity_day = today

        # Per pair prijs ophalen en grid-actie
        for p in pairs:
            t = ex.fetch_ticker(p)
            px = float(t["last"])
            grid = st["grids"][p]
            logs = try_fill_grid(ex, p, px, grid["last_price"], grid, st["portfolio"], st["portfolio"]["start_wallet_eur"])
            if logs:
                print("\n".join(logs))

        # Skim (direct afromen / budget cap handhaven)
        maybe_do_skim(ex, st, pairs)

        now = time.time()
        # Korte summary elke ~10 minuten
        if now - last_sum >= 600:
            eq = mark_to_market(ex, st, pairs)
            cash = avail_eur_for_bot(ex, st, st["portfolio"]["start_wallet_eur"], st["portfolio"]["budget_eur"])
            pr = st["portfolio"]["pnl_realized"]
            print(f"[SUMMARY] equity=€{eq:.2f} | cash=€{cash:.2f} | {fmt_pnl('pnl_realized=', pr)} | "
                  f"skim={C.YELLOW}€{st['portfolio']['skimmed_total']:.2f}{C.RESET}")
            last_sum = now

        # Groot rapport iedere REPORT_EVERY_HOURS
        if now - last_report_ts >= REPORT_EVERY_HOURS * 3600:
            eq = mark_to_market(ex, st, pairs)
            cash = avail_eur_for_bot(ex, st, st["portfolio"]["start_wallet_eur"], st["portfolio"]["budget_eur"])
            pr = st["portfolio"]["pnl_realized"]
            print(f"{C.BOLD}[report]{C.RESET} Equity: €{eq:.2f} | cash=€{cash:.2f} | {fmt_pnl('pnl_realized=', pr)} | "
                  f"budget=€{st['portfolio']['budget_eur']:.2f} | skim={C.YELLOW}€{st['portfolio']['skimmed_total']:.2f}{C.RESET}")
            last_report_ts = now

        save_state(st)
        time.sleep(SLEEP_SEC)

    except ccxt.BaseError as e:
        # veelvoorkomende setup-fouten zichtbaar houden
        msg = str(e)
        if "operatorId" in msg:
            print(f"{C.RED}[err]{C.RESET} Bitvavo vereist 'operatorId'. Zet env OPERATOR_ID, of controleer je account-instellingen.")
        elif "No active API key" in msg:
            print(f"{C.RED}[err]{C.RESET} API key niet actief of geen rechten. Controleer Bitvavo (Lezen + Handel) & Render env.")
        else:
            print(f"[net/ccxt] {e}; backoff..")
        time.sleep(2 + random.random())
    except Exception as e:
        print(f"[runtime] {e}")
        time.sleep(5)
