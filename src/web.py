import requests
import numpy as np
from flask import Flask, jsonify, request
from src.service import build_service


def embed_dialogue(dialogue):
    res = requests.post('http://localhost:5000/context', json=[dialogue])
    result = np.array(res.json()['data'][0])
    return result

# Creating retrieve service
service = build_service(100)

# Creating app instance
app = Flask(__name__)


# Convert context embedding builder
@app.route("/", methods=["POST", "GET"])
def context():
    dialogue = request.get_json(force=True)
    result = service.retrieve(embed_dialogue(dialogue))

    return jsonify({
        "code": 0,
        "message": "ok",
        "data": result
    })
