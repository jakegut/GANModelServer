from flask import Flask, jsonify, request

from generate import Generator
import json

app = Flask(__name__)

gen = Generator("ZeldaGAN/ZeldaFixedDungeonsAlNoDoors_10000_10.pth", 6, 10)

@app.route("/api/get-level", methods=['GET'])
def get_level():
    level = gen.generate()
    return jsonify(level)

@app.route("/api/get-level", methods=['POST'])
def get_level_post():
    data = request.get_json()

    level = [gen.generate(vector=v) for v in data['vectors']]
    return jsonify(level)

if __name__ == "__main__":
    app.run()