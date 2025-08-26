# keep_alive.py
from flask import Flask
import os

_app = Flask(__name__)

@_app.get("/")
def root():
    return "OK", 200

def keep_alive():
    # Alleen starten als Render geen web al draait
    port = int(os.getenv("PORT", "10000"))
    from threading import Thread
    t = Thread(target=_app.run, kwargs={"host":"0.0.0.0","port":port}, daemon=True)
    t.start()

