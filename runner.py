# runner.py — start paper_grid.py en bewaak proces
import subprocess, time, os, pathlib

def heartbeat():
    # toon laatste equity (als aanwezig)
    eq = None
    p = pathlib.Path(os.getenv("DATA_DIR", "/var/data")) / "equity.csv"
    if p.exists():
        try:
            *_, last = p.read_text(encoding="utf-8").strip().splitlines()
            if "," in last:
                eq = last.split(",")[1]
        except: pass
    print(f"[runner] heartbeat… Equity={eq or 'n/a'}")

def main():
    while True:
        print("[runner] start paper_grid.py …")
        proc = subprocess.Popen(["python", "-u", "paper_grid.py"])
        last = time.time()
        try:
            while True:
                time.sleep(10)
                if proc.poll() is not None:
                    print("[runner] paper_grid.py is gestopt; herstart over 5s…")
                    time.sleep(5)
                    break
                if time.time() - last > 300:   # elke 5 min
                    heartbeat()
                    last = time.time()
        except KeyboardInterrupt:
            proc.terminate()
            break

if __name__ == "__main__":
    main()
