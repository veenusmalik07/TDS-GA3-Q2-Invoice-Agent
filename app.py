import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "GA3 Q2 Invoice Agent"
    })

@app.get("/.well-known/agent.json")
def agent_card():
    return jsonify({
        "name": "GA3 Invoice Agent",
        "description": "Evidence-based invoice processing agent",
        "url": os.environ.get("PUBLIC_URL", ""),
        "version": "1.0.0",
        "capabilities": {},
        "skills": []
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
