from flask import Flask
from threading import Thread
import os

app = Flask(__name__)


@app.route("/")
def home():
    return "Bot is running!"


def run():
    port = int(os.environ.get("PORT", 8080))  # gebruik Replit poort of 8080
    app.run(host="0.0.0.0", port=port, debug=False)


def keep_alive():
    t = Thread(target=run, daemon=True)
    t.start()
