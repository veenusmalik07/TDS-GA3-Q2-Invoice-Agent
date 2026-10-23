
import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from threading import RLock
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

AGENT_NAME = "GA3 Invoice Agent"
AGENT_VERSION = "1.0.0"
PUBLIC_URL = os.environ.get(
    "PUBLIC_URL", "https://tds-ga3-q2-invoice-agent.onrender.com"
).rstrip("/")
API_TOKEN = os.environ.get("API_TOKEN", "")
DATABASE = os.environ.get("DATABASE_PATH", "/tmp/ga3_tasks.db")

INPUT_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MODE = "application/vnd.ga5.invoice-action-receipts+json"
RESULT_MODE = "application/vnd.ga5.invoice-action-results+json"
A2A_VERSION = "1.0"

lock = RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def db():
    conn = sqlite3.connect(DATABASE, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                principal TEXT NOT NULL,
                context_id TEXT NOT NULL,
                batch_id TEXT,
                state TEXT NOT NULL,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                principal TEXT NOT NULL,
                message_id TEXT NOT NULL,
                message_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (principal, message_id)
            )
        """)


init_db()


def json_response(data, status=200, media_type="application/a2a+json"):
    return Response(
        json.dumps(data, separators=(",", ":"), ensure_ascii=False),
        status=status,
        content_type=f"{media_type}; charset=utf-8",
    )


def error(code, message, status):
    return json_response({"error": {"code": code, "message": message}}, status)


def principal():
    header = request.headers.get("Authorization", "")
    if not API_TOKEN:
        return None
    if not header.startswith("Bearer "):
        return None
    token = header[7:].strip()
    if not token or not hashlib.sha256(token.encode()).digest() == hashlib.sha256(
        API_TOKEN.encode()
    ).digest():
        return None
    return hashlib.sha256(token.encode()).hexdigest()


def authenticated():
    user = principal()
    if user is None:
        return None, error("UNAUTHENTICATED", "Bearer authentication required", 401)
    return user, None


def check_protocol():
    version = request.headers.get("A2A-Version")
    if version != A2A_VERSION:
        return error("UNSUPPORTED_VERSION", "Require A2A-Version: 1.0", 400)

    content_type = request.headers.get("Content-Type", "").split(";")[0].strip()
    if content_type != "application/a2a+json":
        return error("UNSUPPORTED_MEDIA_TYPE", "Require application/a2a+json", 415)
    return None


def task_view(task):
    return {
        "id": task["id"],
        "contextId": task["contextId"],
        "status": task["status"],
        "history": task["history"],
        "artifacts": task.get("artifacts", []),
        "metadata": task.get("metadata", {}),
    }


def get_task(task_id, user):
    with db() as conn:
        row = conn.execute(
            "SELECT task_json FROM tasks WHERE id=? AND principal=?",
            (task_id, user),
        ).fetchone()
    if not row:
        return None
    return json.loads(row["task_json"])


def save_task(task, user, batch_id=None):
    timestamp = now()
    with db() as conn:
        conn.execute("""
            INSERT INTO tasks
                (id, principal, context_id, batch_id, state, task_json,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                state=excluded.state,
                task_json=excluded.task_json,
                updated_at=excluded.updated_at
        """, (
            task["id"], user, task["contextId"], batch_id,
            task["status"]["state"], json.dumps(task),
            timestamp, timestamp
        ))


def canonical_hash(message):
    canonical = json.dumps(
        message, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def get_saved_message(user, message_id):
    with db() as conn:
        return conn.execute(
            "SELECT message_hash, task_id FROM messages "
            "WHERE principal=? AND message_id=?",
            (user, message_id),
        ).fetchone()


def remember_message(user, message_id, message_hash, task_id):
    with db() as conn:
        conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            (user, message_id, message_hash, task_id),
        )


def make_task(task_id, context_id, message):
    return {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": "TASK_STATE_INPUT_REQUIRED",
            "timestamp": now(),
        },
        "history": [message],
        "artifacts": [],
        "metadata": {},
    }


def get_message_data(message):
    for part in message.get("parts", []):
        if part.get("mediaType") == INPUT_MODE:
            return part.get("data")
    return None


def build_proposals(batch):
    """
    Safe placeholder decision layer.

    Replace this function with the assignment's AI decision step.
    It deliberately does not invent evidence or approve payments without
    verifying the invoice documents and policy.
    """
    proposals = []

    for package in batch.get("packages", []):
        package_id = package.get("packageId") or package.get("id")
        if not package_id:
            raise ValueError("Each package must have a packageId or id")

        # Until the invoice evidence and policy have been evaluated,
        # pause payment and request verification.
        proposals.append({
            "packageId": package_id,
            "actionId": hashlib.sha256(
                f'{batch.get("batchId")}:{package_id}'.encode()
            ).hexdigest()[:24],
            "action": "hold_invoice",
            "facts": {
                "vendorName": "",
                "invoiceNumber": "",
                "amountMinor": 0,
                "currency": "INR",
            },
            "evidenceRefs": [],
            "rationale": (
                "hold_invoice: the invoice evidence and policy must be "
                "verified before a safe payment decision can be made."
            ),
        })

    return {
        "batchId": batch.get("batchId", ""),
        "proposals": proposals,
    }


def process_initial_message(message, user, configuration):
    message_id = message.get("messageId")
    context_id = message.get("contextId") or message_id
    if not message_id:
        return error("INVALID_MESSAGE", "messageId is required", 400)

    message_hash = canonical_hash(message)
    previous = get_saved_message(user, message_id)

    if previous:
        if previous["message_hash"] != message_hash:
            return error(
                "IDEMPOTENCY_CONFLICT",
                "Message ID was reused with different content",
                409,
            )
        task = get_task(previous["task_id"], user)
        if task is None:
            return error("TASK_NOT_FOUND", "Task not found", 404)
        return json_response({"task": task_view(task)})

    batch = get_message_data(message)
    if not isinstance(batch, dict):
        return error("INVALID_MESSAGE", "Invoice batch data is missing", 400)

    batch_id = batch.get("batchId")
    if not batch_id or not isinstance(batch.get("packages"), list):
        return error(
            "INVALID_BATCH", "batchId and packages are required", 400
        )

    task_id = hashlib.sha256(
        f"{user}:{context_id}:{message_id}".encode()
    ).hexdigest()[:32]

    try:
        proposal_data = build_proposals(batch)
    except (ValueError, TypeError) as exc:
        return error("INVALID_BATCH", str(exc), 400)

    task = make_task(task_id, context_id, message)
    task["status"] = {
        "state": "TASK_STATE_INPUT_REQUIRED",
        "timestamp": now(),
    }
    task["artifacts"] = [{
        "artifactId": f"proposal-{task_id}",
        "parts": [{
            "mediaType": PROPOSAL_MODE,
            "data": proposal_data,
        }],
    }]

    # Persist before replying so retries return the same Task.
    with lock:
        save_task(task, user, batch_id)
        remember_message(user, message_id, message_hash, task_id)

    return json_response({"task": task_view(task)})


@app.get("/")
def health():
    return jsonify({"service": AGENT_NAME, "status": "ok"})


@app.get("/.well-known/agent-card.json")
def agent_card():
    base_url = f"{PUBLIC_URL}/a2a/"
    return jsonify({
        "name": AGENT_NAME,
        "description": "Evidence-based invoice processing agent",
        "version": AGENT_VERSION,
        "url": base_url,
        "capabilities": {},
        "defaultInputModes": [INPUT_MODE],
        "defaultOutputModes": [PROPOSAL_MODE, RECEIPT_MODE],
        "supportedInterfaces": [{
            "url": base_url,
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": A2A_VERSION,
        }],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": "Reviews invoice evidence and proposes safe actions",
            "tags": ["invoice", "payments", "evidence", "approval"],
        }],
    })


@app.post("/a2a/message:send")
def send_message():
    protocol_error = check_protocol()
    if protocol_error:
        return protocol_error

    user, auth_error = authenticated()
    if auth_error:
        return auth_error

    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        return error("INVALID_ENVELOPE", "A message object is required", 400)

    message = body["message"]
    if message.get("role") != "ROLE_USER":
        return error("INVALID_MESSAGE", "Expected ROLE_USER", 400)

    # Initial invoice batch message.
    if get_message_data(message) is not None:
        return process_initial_message(
            message, user, body.get("configuration", {})
        )

    # Results continuations require strict validation against stored proposals
    # and real receipt nonces. Do not execute unvalidated continuations.
    return error(
        "CONTINUATION_NOT_IMPLEMENTED",
        "Result continuation handling must be implemented before grading",
        501,
    )


@app.get("/a2a/tasks")
def list_tasks():
    protocol_error = check_protocol()
    if protocol_error:
        return protocol_error

    user, auth_error = authenticated()
    if auth_error:
        return auth_error

    with db() as conn:
        rows = conn.execute(
            "SELECT task_json FROM tasks WHERE principal=? ORDER BY created_at DESC",
            (user,),
        ).fetchall()

    return json_response({
        "tasks": [task_view(json.loads(row["task_json"])) for row in rows]
    })


@app.get("/a2a/tasks/<task_id>")
def read_task(task_id):
    protocol_error = check_protocol()
    if protocol_error:
        return protocol_error

    user, auth_error = authenticated()
    if auth_error:
        return auth_error

    task = get_task(task_id, user)
    if task is None:
        return error("TASK_NOT_FOUND", "Task not found", 404)

    return json_response({"task": task_view(task)})


@app.post("/a2a/tasks/<task_id>:cancel")
def cancel_task(task_id):
    protocol_error = check_protocol()
    if protocol_error:
        return protocol_error

    user, auth_error = authenticated()
    if auth_error:
        return auth_error

    with lock:
        task = get_task(task_id, user)
        if task is None:
            return error("TASK_NOT_FOUND", "Task not found", 404)

        state = task["status"]["state"]
        if state in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            return error("TASK_TERMINAL", "Task is already terminal", 409)

        task["status"] = {
            "state": "TASK_STATE_CANCELED",
            "timestamp": now(),
        }
        save_task(task, user, task.get("batchId"))

    return json_response({"task": task_view(task)})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
    )
