import os
from flask import Flask, jsonify, request

app = Flask(__name__)

AGENT_NAME = "GA3 Invoice Agent"
AGENT_VERSION = "1.0.0"
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
API_TOKEN = os.environ.get("API_TOKEN", "")


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": AGENT_NAME
    })


@app.get("/.well-known/agent-card.json")
def agent_card():
    return jsonify({
        "name": AGENT_NAME,
        "description": "Evidence-based invoice processing agent",
        "version": AGENT_VERSION,
        "url": f"{PUBLIC_URL}/a2a/",
        "capabilities": {},
        "defaultInputModes": [
            "application/vnd.ga5.invoice-claim-batch+json"
        ],
        "defaultOutputModes": [
            "application/vnd.ga5.invoice-action-proposals+json",
            "application/vnd.ga5.invoice-action-receipts+json"
        ],
        "supportedInterfaces": [
            {
                "url": f"{PUBLIC_URL}/a2a/",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0"
            }
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": "Reviews invoice evidence and proposes safe actions",
                "tags": ["invoice", "payments", "evidence", "approval"]
            }
        ]
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
