from flask import Flask, jsonify

from generate import generate
import json

app = Flask(__name__)

@app.route("/api/get-level")
def get_level():
    level = generate("ZeldaGAN/ZeldaFixedDungeonsAlNoDoors_10000_10.pth", 6, 10)
    return jsonify(level)

if __name__ == "__main__":
    app.run()