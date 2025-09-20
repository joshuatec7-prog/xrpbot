def try_grid_live(ex, pair, price_now, price_prev, state, grid, pairs) -> List[str]:
    levels = grid["levels"]
    port = state["portfolio"]
    logs = []
    if not levels:
        return logs

    # helper voor TP/SL berekenen
    def tp_price(buy_px: float) -> float:
        # zelfde drempel als sell: winst + fees + safety
        return buy_px * (1.0 + MIN_PROFIT_PCT + 2.0 * FEE_PCT + SELL_SAFETY_PCT)

    def active_sl(lot) -> float:
        hard_stop = lot["buy_price"] * (1.0 - STOP_LOSS_PCT) if STOP_LOSS_PCT > 0 else 0.0
        trail_stop = lot.get("peak", lot["buy_price"]) * (1.0 - TRAIL_STOP_PCT) if TRAIL_STOP_PCT > 0 else 0.0
        return max(hard_stop, trail_stop)

    def bid_px() -> float:
        try:
            ob = ex.fetch_order_book(pair, limit=5)
            if ob and ob.get("bids"):
                return float(ob["bids"][0][0])
        except Exception:
            pass
        t = ex.fetch_ticker(pair)
        return float(t.get("bid") or t.get("last"))

    # Circuit breaker: pauze BUY’s bij drawdown
    if MAX_DRAWDOWN_PCT > 0:
        since_start_pct = (mark_to_market(ex, state, pairs) - CAPITAL_EUR) / max(1.0, CAPITAL_EUR)
        if since_start_pct <= -MAX_DRAWDOWN_PCT:
            logs.append(f"{COL_R}[{pair}] BUY skip: circuit-breaker (drawdown {since_start_pct:.1%}).{COL_RESET}")
            crossed = []
        else:
            crossed = [L for L in levels if (price_prev is not None and price_now < price_prev and price_now < L <= price_prev)]
    else:
        crossed = [L for L in levels if (price_prev is not None and price_now < price_prev and price_now < L <= price_prev)]

    # BUY
    avail = free_eur_on_exchange(ex)
    live_cap = bot_inventory_value_eur_from_exchange(ex, state, pairs)
    cap = cap_now(state)

    for _ in crossed:
        ticket = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels))
        max_cost = max(0.0, avail - MIN_CASH_BUFFER_EUR) / (1.0 + FEE_PCT)
        cost = min(ticket, max_cost)

        min_quote, min_base = market_mins(ex, pair, price_now)
        required = max(min_quote, (min_base * price_now))
        cost = max(cost, required)
        cost = math.ceil(cost)
        cost = math.ceil(cost * 1.01)

        if invested_cost_eur(state) + cost > cap + 1e-6:
            logs.append(f"{COL_C}[{pair}] BUY skip: cost-cap (cap=€{cap:.2f}).{COL_RESET}")
            continue
        if live_cap + cost > cap + 1e-6:
            logs.append(f"{COL_C}[{pair}] BUY skip: live-cap (≈€{live_cap:.2f}/{cap:.2f}).{COL_RESET}")
            continue

        qty, avg, fee, executed = buy_market(ex, pair, cost)
        if qty <= 0 or avg <= 0:
            continue
        if qty < min_base or executed < min_quote:
            logs.append(
                f"{COL_C}[{pair}] BUY fill < minima (amt={qty:.8f} < {min_base} of €{executed:.2f} < €{min_quote:.2f}); overslaan.{COL_RESET}"
            )
            continue

        # lot opslaan + peak voor trailing
        lot = {"qty": qty, "buy_price": avg, "peak": avg}
        grid["inventory_lots"].append(lot)
        port["cash_eur"] -= (executed + fee)
        port["coins"][pair]["qty"] += qty
        avail -= (executed + fee)
        live_cap += executed

        append_csv(
            TRADES_CSV,
            [now_iso(), pair, "BUY", f"{avg:.6f}", f"{qty:.8f}", f"{executed:.2f}", f"{port['cash_eur']:.2f}",
             pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
            header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"],
        )
        # >>> NEW: duidelijk plan loggen
        tp = tp_price(avg)
        sl = active_sl(lot)
        logs.append(
            f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avg:.6f} | PLAN: TP ≥ €{tp:.6f}"
            + (f" | SL ≤ €{sl:.6f}" if sl > 0 else " | SL (uit)")
            + f" | now €{price_now:.6f}{COL_RESET}"
        )

    # SELL
    if grid["inventory_lots"]:
        base = pair.split("/")[0]
        bot_free = None
        if LOCK_PREEXISTING_BALANCE and "baseline" in state:
            bot_free = max(0.0, free_base_on_exchange(ex, base) - float(state["baseline"].get(base, 0.0)))

        min_quote, min_base = market_mins(ex, pair, price_now)

        # >>> NEW: Toon per lot het verkoop-plan (kort)
        bpx = bid_px()
        for i, l in enumerate(grid["inventory_lots"][:3]):  # max 3 regels per cycle om spam te beperken
            l["peak"] = max(l.get("peak", l["buy_price"]), price_now)
            tp = tp_price(l["buy_price"])
            sl = active_sl(l)
            logs.append(
                f"[{pair}] lot#{i+1} PLAN | buy €{l['buy_price']:.6f} | peak €{l['peak']:.6f} | TP ≥ €{tp:.6f}"
                + (f" | SL ≤ €{sl:.6f}" if sl > 0 else " | SL (uit)")
                + f" | now €{price_now:.6f} | bid €{bpx:.6f}"
            )

        changed = True
        while changed and grid["inventory_lots"]:
            changed = False
            lot = grid["inventory_lots"][0]

            # update trailing peak
            lot["peak"] = max(lot.get("peak", lot["buy_price"]), price_now)

            # STOP-LOSS
            hard_stop_px = lot["buy_price"] * (1.0 - STOP_LOSS_PCT) if STOP_LOSS_PCT > 0 else 0.0
            trail_stop_px = lot["peak"] * (1.0 - TRAIL_STOP_PCT) if TRAIL_STOP_PCT > 0 else 0.0
            active_stop_px = max(hard_stop_px, trail_stop_px)

            if active_stop_px > 0 and price_now <= active_stop_px:
                stop_qty = amount_to_precision(ex, pair, lot["qty"])
                if stop_qty >= min_base and (stop_qty * price_now) >= min_quote:
                    proceeds, avg, fee = sell_market(ex, pair, stop_qty)
                    if proceeds > 0 and avg > 0:
                        grid["inventory_lots"].pop(0)
                        pnl = proceeds - fee - (stop_qty * lot["buy_price"])
                        port["cash_eur"] += (proceeds - fee)
                        port["coins"][pair]["qty"] -= stop_qty
                        port["pnl_realized"] += pnl
                        if bot_free is not None:
                            bot_free -= stop_qty
                        append_csv(
                            TRADES_CSV,
                            [now_iso(), pair, "SELL", f"{avg:.6f}", f"{stop_qty:.8f}", f"{proceeds:.2f}", f"{port['cash_eur']:.2f}",
                             base, f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "stop_loss"],
                        )
                        logs.append(f"{COL_R}[{pair}] STOP-LOSS {stop_qty:.8f} @ €{avg:.6f} | pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}")
                        changed = True
                        continue

            # Take-profit
            idx = next(
                (i for i, l in enumerate(grid["inventory_lots"])
                 if net_gain_ok(l["buy_price"], price_now, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, l["qty"])),
                None,
            )
            if idx is None:
                break

            lot = grid["inventory_lots"][idx]
            qty = lot["qty"]

            if qty < min_base or (qty * price_now) < min_quote:
                logs.append(f"[{pair}] SELL skip: lot te klein (amt {qty:.8f} / min {min_base} of €{qty*price_now:.2f} / min €{min_quote:.2f}).")
                break

            if bot_free is not None and bot_free + 1e-12 < qty:
                logs.append(f"[{pair}] SELL stop: baseline-protect ({bot_free:.8f} {base} beschikbaar).")
                break

            bpx = bid_px()
            trigger_px = lot["buy_price"] * (1.0 + MIN_PROFIT_PCT + 2.0*FEE_PCT + SELL_SAFETY_PCT)
            if bpx + 1e-12 < trigger_px:
                logs.append(f"[{pair}] SELL wait: bid €{bpx:.6f} < trigger €{trigger_px:.6f}.")
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
                     base, f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "take_profit"],
                )
                col = COL_G if pnl >= 0 else COL_R
                logs.append(
                    f"{col}[{pair}] SELL {sell_qty:.8f} @ €{avg:.6f} | TP bereikt | pnl=€{pnl:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}"
                )
                changed = True

    grid["last_price"] = price_now
    return logs
