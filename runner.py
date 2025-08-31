# runner.py
# Start de paper grid bot en draait elke 12 uur een rapport

import subprocess
import time
import threading

# --- Start de paper grid bot ---
def start_bot():
    print(">> Start paper grid bot...")
    subprocess.Popen(["python", "paper_grid.py"])

# --- Start rapportage elke 12 uur ---
def run_report():
    while True:
        print(">> Genereer rapport (report_paper.py)...")
        try:
            subprocess.run(["python", "python_report_paper.py"], check=True)
        except Exception as e:
            print(f"Fout tijdens rapportage: {e}")
        time.sleep(12 * 3600)  # wacht 12 uur

if __name__ == "__main__":
    # Start bot in aparte thread
    threading.Thread(target=start_bot, daemon=True).start()
    # Start rapportage-loop
    run_report()
