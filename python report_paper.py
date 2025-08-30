# report_paper.py
# Analyseert je trades.csv en geeft winst/verlies overzicht

import pandas as pd
from pathlib import Path

TRADES_FILE = Path("trades.csv")

if not TRADES_FILE.exists():
    print("⚠️ Geen trades.csv gevonden.")
    exit()

df = pd.read_csv(TRADES_FILE)

# Convert kolommen
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
df["pnl_eur"] = pd.to_numeric(df.get("pnl_eur", 0), errors="coerce")

# Basis stats
print("\n=== 📊 TRADE RAPPORT ===")
print(f"Totaal trades: {len(df)}")

total_pnl = df["pnl_eur"].sum()
print(f"Totale PnL: €{total_pnl:.2f}")

wins = df[df["pnl_eur"] > 0]
losses = df[df["pnl_eur"] < 0]

print(f"Winrate: {len(wins)}/{len(df)} ({(len(wins)/len(df)*100 if len(df)>0 else 0):.1f}%)")
print(f"Grootste winst trade: €{wins['pnl_eur'].max():.2f}" if not wins.empty else "Geen winsten")
print(f"Grootste verlies trade: €{losses['pnl_eur'].min():.2f}" if not losses.empty else "Geen verliezen")

# Gemiddelde PnL per trade
if len(df) > 0:
    print(f"Gemiddelde PnL/trade: €{total_pnl/len(df):.2f}")

# Laatste 5 trades
print("\n--- Laatste 5 trades ---")
print(df.tail(5).to_string(index=False))
