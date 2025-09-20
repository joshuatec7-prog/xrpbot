def try_grid_live(ex, pair, price_now, price_prev, state, grid, pairs) -> list[str]:
    levels = grid["levels"]
    port   = state["portfolio"]
    logs: list[str] = []
    if not levels:
        return logs

    # ---------- BUY ----------
    if price_prev is not None and price_now < price_prev:
        crossed = [L for L in levels if price_now < L <= price_prev]
        avail   = free_eur_on_exchange(ex)
        live_cap = bot_inventory_value_eur_from_exchange(ex, state, pairs)
        cap     = cap_now(state)

        for L in crossed:
            # hoeveel zouden we kopen bij dit level?
            ticket_eur = euro_per_ticket(port["coins"][pair]["cash_alloc"], len(levels))
            max_cost   = max(0.0, avail - MIN_CASH_BUFFER_EUR) / (1.0 + FEE_PCT)
            cost       = min(ticket_eur, max_cost)

            # minima (quote/base)
            min_quote, min_base = market_mins(ex, pair, price_now)
            required = max(min_quote, (min_base * price_now))
            cost     = max(cost, required)
            # afronden naar boven, en een tikkeltje marge
            cost     = math.ceil(cost)
            cost     = math.ceil(cost * 1.01)

            # Als we NU zouden kopen op ~level L, wat is dan de verkooptarget?
            plan_target = target_sell_price(L)

            # caps check
            if invested_cost_eur(state) + cost > cap + 1e-6:
                logs.append(
                    f"[{pair}] BUY skip: cost-cap bereikt (cap=€{cap:.2f}). "
                    f"PLAN: koop @≈€{L:.2f} → target SELL≈€{plan_target:.2f}"
                )
                continue
            if live_cap + cost > cap + 1e-6:
                logs.append(
                    f"[{pair}] BUY skip: live-cap (≈€{live_cap:.2f}/{cap:.2f}). "
                    f"PLAN: koop @≈€{L:.2f} → target SELL≈€{plan_target:.2f}"
                )
                continue

            # Geld te krap? Laat exact zien wat nodig is en de targets.
            if avail < required + MIN_CASH_BUFFER_EUR:
                tekort = (required + MIN_CASH_BUFFER_EUR) - avail
                logs.append(
                    f"[{pair}] BUY skip: vrije EUR te laag (free≈€{avail:.2f}, nodig≈€{required+MIN_CASH_BUFFER_EUR:.2f}, tekort≈€{tekort:.2f}). "
                    f"PLAN: koop @≈€{L:.2f} → target SELL≈€{plan_target:.2f}"
                )
                continue

            # Probeer marktkoop
            qty, avg, fee, executed = buy_market(ex, pair, cost)
            if qty <= 0 or avg <= 0:
                logs.append(
                    f"[{pair}] BUY fail: geen fill. PLAN was: koop @≈€{L:.2f} → target SELL≈€{plan_target:.2f}"
                )
                continue

            if qty < min_base or executed < min_quote:
                logs.append(
                    f"[{pair}] BUY fill < minima (amt={qty:.8f} < {min_base} of €{executed:.2f} < €{min_quote:.2f}); overslaan. "
                    f"PLAN: koop @≈€{L:.2f} → target SELL≈€{plan_target:.2f}"
                )
                continue

            # Bewaren als lot
            grid["inventory_lots"].append({"qty": qty, "buy_price": avg})
            port["cash_eur"] -= (executed + fee)
            port["coins"][pair]["qty"] += qty
            avail   -= (executed + fee)
            live_cap += executed

            target_now = target_sell_price(avg)
            append_csv(
                TRADES_CSV,
                [now_iso(), pair, "BUY", f"{avg:.6f}", f"{qty:.8f}", f"{executed:.2f}", f"{port['cash_eur']:.2f}",
                 pair.split("/")[0], f"{port['coins'][pair]['qty']:.8f}", "", "grid_buy"],
                header=["timestamp","pair","side","avg_price","qty","eur","cash_eur","base","base_qty","pnl_eur","comment"]
            )
            logs.append(
                f"{COL_C}[{pair}] BUY {qty:.8f} @ €{avg:.6f} | req≈€{cost:.2f} | exec≈€{executed:.2f} | fee≈€{fee:.2f} | "
                f"→ target SELL≈€{target_now:.2f} | cash=€{port['cash_eur']:.2f}{COL_RESET}"
            )

    # ---------- SELL ----------
    if grid["inventory_lots"]:
        base = pair.split("/")[0]
        bot_free = None
        if LOCK_PREEXISTING_BALANCE and "baseline" in state:
            bot_free = max(0.0, free_base_on_exchange(ex, base) - float(state["baseline"].get(base, 0.0)))

        min_quote, min_base = market_mins(ex, pair, price_now)

        changed = True
        while changed and grid["inventory_lots"]:
            changed = False
            # Is er een lot dat aan de netto-winstdrempel voldoet?
            idx = next(
                (i for i, l in enumerate(grid["inventory_lots"])
                 if net_gain_ok(l["buy_price"], price_now, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, l["qty"])),
                None
            )
            if idx is None:
                # Toon voor de bovenste lot alvast de actuele 'wait' met trigger
                if grid["inventory_lots"]:
                    top = grid["inventory_lots"][0]
                    bid_px   = best_bid_px(ex, pair)
                    trigger  = target_sell_price(top["buy_price"])
                    logs.append(
                        f"[{pair}] SELL wait: bid €{bid_px:.2f} < trigger €{trigger:.2f} "
                        f"(buy €{top['buy_price']:.2f})"
                    )
                break

            lot = grid["inventory_lots"][idx]
            qty = lot["qty"]

            if qty < min_base or (qty * price_now) < min_quote:
                logs.append(
                    f"[{pair}] SELL skip: lot te klein (amt {qty:.8f} / min {min_base} of €{qty*price_now:.2f} / min €{min_quote:.2f}). "
                    f"(buy €{lot['buy_price']:.2f} → trigger €{target_sell_price(lot['buy_price']):.2f})"
                )
                break

            if bot_free is not None and bot_free + 1e-12 < qty:
                logs.append(
                    f"[{pair}] SELL stop: baseline-protect ({bot_free:.8f} {base} beschikbaar). "
                    f"(buy €{lot['buy_price']:.2f})"
                )
                break

            # Extra veiligheid op best bid
            bid_px  = best_bid_px(ex, pair)
            trigger = target_sell_price(lot["buy_price"])
            if bid_px + 1e-12 < trigger:
                logs.append(
                    f"[{pair}] SELL wait: bid €{bid_px:.2f} < trigger €{trigger:.2f} "
                    f"(buy €{lot['buy_price']:.2f})"
                )
                break

            # Verkoop
            sell_qty = amount_to_precision(ex, pair, qty)
            if sell_qty <= 0 or sell_qty + 1e-15 < min_base:
                logs.append(f"[{pair}] SELL skip: qty {sell_qty:.8f} < min {min_base}."); break

            proceeds, avg, fee = sell_market(ex, pair, sell_qty)
            if proceeds > 0 and avg > 0 and net_gain_ok(lot["buy_price"], avg, FEE_PCT, MIN_PROFIT_PCT, MIN_PROFIT_EUR, sell_qty):
                grid["inventory_lots"].pop(idx)
                pnl = proceeds - fee - (sell_qty * lot["buy_price"])
                port["cash_eur"]         += (proceeds - fee)
                port["coins"][pair]["qty"] -= sell_qty
                port["pnl_realized"]      += pnl
                if bot_free is not None:
                    bot_free -= sell_qty

                append_csv(
                    TRADES_CSV,
                    [now_iso(), pair, "SELL", f"{avg:.6f}", f"{sell_qty:.8f}", f"{proceeds:.2f}", f"{port['cash_eur']:.2f}",
                     base, f"{port['coins'][pair]['qty']:.8f}", f"{pnl:.2f}", "take_profit"]
                )
                col = COL_G if pnl >= 0 else COL_R
                logs.append(
                    f"{col}[{pair}] SELL {sell_qty:.8f} @ €{avg:.6f} | proceeds=€{proceeds:.2f} | fee=€{fee:.2f} | "
                    f"pnl=€{pnl:.2f} | (buy €{lot['buy_price']:.2f} → trigger €{trigger:.2f}) | cash=€{port['cash_eur']:.2f}{COL_RESET}"
                )
                changed = True

    grid["last_price"] = price_now
    return logs
