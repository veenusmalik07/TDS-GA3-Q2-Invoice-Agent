import os
import json
import uuid
import hashlib
import hmac
import sqlite3
import threading
from datetime import datetime, timezone

from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

AGENT_NAME = "GA3 Invoice Agent"
AGENT_VERSION = "1.0.0"

PUBLIC_URL = os.environ.get(
    "PUBLIC_URL",
    "https://tds-ga3-q2-invoice-agent.onrender.com"
).rstrip("/")

API_TOKEN = os.environ.get("API_TOKEN", "")
DATABASE = os.environ.get("DATABASE_PATH", "agent_tasks.db")

INPUT_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MODE = "application/vnd.ga5.invoice-action-receipts+json"
RESULT_MODE = "application/vnd.ga5.invoice-action-results+json"
A2A_JSON = "application/a2a+json"

ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}

DB_LOCK = threading.RLock()


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def connect_db():
    conn = sqlite3.connect(DATABASE, timeout=20)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with DB_LOCK:
        conn = connect_db()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                context_id TEXT NOT NULL,
                owner TEXT NOT NULL,
                status TEXT NOT NULL,
                batch_id TEXT,
                task_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                owner TEXT NOT NULL,
                message_id TEXT NOT NULL,
                semantic_hash TEXT NOT NULL,
                task_id TEXT NOT NULL,
                PRIMARY KEY (owner, message_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                task_id TEXT NOT NULL,
                package_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                PRIMARY KEY (task_id, package_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS receipts (
                task_id TEXT NOT NULL,
                package_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                receipt_nonce TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                PRIMARY KEY (task_id, package_id)
            )
        """)

        conn.commit()
        conn.close()


init_db()


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def json_response(data, status=200, media_type=A2A_JSON):
    response = make_response(jsonify(data), status)
    response.headers["Content-Type"] = f"{media_type}; charset=utf-8"
    response.headers["A2A-Version"] = "1.0"
    return response


def error(message, status=400, code="INVALID_REQUEST"):
    return json_response({
        "error": {
            "code": code,
            "message": message
        }
    }, status)


# ---------------------------------------------------------
# Authentication
# ---------------------------------------------------------

def extract_bearer():
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")

    if scheme.lower() != "bearer" or not token:
        return None

    return token


def get_owner():
    token = extract_bearer()

    if not API_TOKEN or not token:
        return None

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def require_owner():
    token = extract_bearer()

    if not API_TOKEN:
        return None, error(
            "Server API_TOKEN is not configured.",
            500,
            "SERVER_CONFIGURATION_ERROR"
        )

    if not token:
        return None, error(
            "Bearer authentication required.",
            401,
            "UNAUTHENTICATED"
        )

    expected = hashlib.sha256(API_TOKEN.encode("utf-8")).hexdigest()
    supplied = hashlib.sha256(token.encode("utf-8")).hexdigest()

    if not hmac.compare_digest(supplied, expected):
        return None, error(
            "Invalid credentials.",
            403,
            "FORBIDDEN"
        )

    return supplied, None


# ---------------------------------------------------------
# A2A protocol helpers
# ---------------------------------------------------------

def check_protocol_headers():
    version = request.headers.get("A2A-Version")

    if version != "1.0":
        return error(
            "A2A-Version: 1.0 is required.",
            400,
            "VERSION_MISMATCH"
        )

    content_type = (
        request.headers.get("Content-Type", "")
        .split(";")[0]
        .strip()
    )

    if content_type not in ("application/json", A2A_JSON):
        return error(
            "Use application/a2a+json or application/json.",
            415,
            "UNSUPPORTED_MEDIA_TYPE"
        )

    return None


def get_message_parts(message):
    parts = message.get("parts")

    if not isinstance(parts, list):
        return []

    return parts


def find_artifact_data(task, media_type):
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("mediaType") == media_type:
                return part.get("data")

    return None


def save_task(
    conn,
    task_id,
    context_id,
    owner,
    status,
    batch_id,
    task
):
    timestamp = now()

    conn.execute("""
        INSERT INTO tasks (
            task_id,
            context_id,
            owner,
            status,
            batch_id,
            task_json,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            status=excluded.status,
            task_json=excluded.task_json,
            updated_at=excluded.updated_at
    """, (
        task_id,
        context_id,
        owner,
        status,
        batch_id,
        json.dumps(task, separators=(",", ":")),
        timestamp,
        timestamp
    ))


def load_task(conn, task_id, owner):
    row = conn.execute(
        "SELECT * FROM tasks WHERE task_id=? AND owner=?",
        (task_id, owner)
    ).fetchone()

    if row is None:
        return None

    return row, json.loads(row["task_json"])


def task_envelope(task):
    return {"task": task}


# ---------------------------------------------------------
# Agent card and health check
# ---------------------------------------------------------

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
        "defaultInputModes": [INPUT_MODE],
        "defaultOutputModes": [
            PROPOSAL_MODE,
            RECEIPT_MODE
        ],
        "supportedInterfaces": [{
            "url": f"{PUBLIC_URL}/a2a/",
            "protocolBinding": "HTTP+JSON",
            "protocolVersion": "1.0"
        }],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice Action Agent",
            "description": (
                "Reviews invoice evidence and proposes safe actions"
            ),
            "tags": [
                "invoice",
                "payments",
                "evidence",
                "approval"
            ]
        }]
    })


# ---------------------------------------------------------
# Invoice decision logic
# ---------------------------------------------------------

def package_documents(package):
    """Collect text values from the supplied invoice package."""
    values = []

    def walk(value, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, str(k))

        elif isinstance(value, list):
            for item in value:
                walk(item, key)

        elif isinstance(value, str):
            values.append((key, value))

    walk(package)
    return values


def decide_package(package, policy_revision=None):
    """
    Conservative evidence-based baseline.

    This baseline does not replace a full document-understanding
    model. It uses the information present in the submitted package.
    """

    package_id = str(package.get("packageId", "")).strip()

    if not package_id:
        package_id = "pkg-" + canonical_hash(package)[:20]

    documents = package_documents(package)
    combined = "\n".join(
        value for _, value in documents
    ).lower()

    vendor = str(package.get("vendorName", "")).strip()
    invoice_number = str(package.get("invoiceNumber", "")).strip()
    amount_minor = package.get("amountMinor")
    currency = str(package.get("currency", "INR")).strip()

    evidence_refs = package.get("evidenceRefs", [])

    if not isinstance(evidence_refs, list):
        evidence_refs = []

    # Extract explicit bracketed references from source text.
    import re

    explicit_refs = []

    for _, value in documents:
        explicit_refs.extend(
            re.findall(r"\[([^\[\]]+)\]", value)
        )

    decisive_refs = list(dict.fromkeys(explicit_refs))[:3]

    # Supplement from explicitly supplied evidence references.
    if len(decisive_refs) < 3:
        for ref in evidence_refs:
            if (
                isinstance(ref, str)
                and ref not in decisive_refs
            ):
                decisive_refs.append(ref)

            if len(decisive_refs) == 3:
                break

    # Default to exception unless evidence supports a safer choice.
    action = "open_exception"
    reasons = [
        "Insufficient verified evidence to authorize an automatic payment."
    ]

    duplicate = any(
        phrase in combined
        for phrase in (
            "already paid",
            "payment completed",
            "previously paid",
            "duplicate invoice",
            "duplicate payment"
        )
    )

    conflicting = any(
        phrase in combined
        for phrase in (
            "conflicting amount",
            "amount mismatch",
            "vendor mismatch",
            "invoice number mismatch",
            "records conflict"
        )
    )

    verification_pending = any(
        phrase in combined
        for phrase in (
            "verification pending",
            "awaiting verification",
            "not verified",
            "verification required",
            "on hold"
        )
    )

    approval_required = any(
        phrase in combined
        for phrase in (
            "approval required",
            "manager approval",
            "outside delegated authority",
            "approval limit exceeded"
        )
    )

    if duplicate:
        action = "reject_duplicate"
        reasons = [
            "Source documents explicitly indicate prior payment or duplication."
        ]

    elif conflicting:
        action = "open_exception"
        reasons = [
            "Source records contain a material conflict."
        ]

    elif verification_pending:
        action = "hold_invoice"
        reasons = [
            "A stated verification requirement remains unresolved."
        ]

    elif approval_required:
        action = "request_approval"
        reasons = [
            "Source documents indicate that approval is required."
        ]

    elif (
        vendor
        and invoice_number
        and isinstance(amount_minor, int)
        and amount_minor > 0
        and len(decisive_refs) >= 2
    ):
        action = "request_approval"
        reasons = [
            "Basic invoice fields and evidence references exist, but "
            "delegated authority and reconciliation have not been "
            "independently verified."
        ]

    action_id = str(uuid.uuid4())

    facts = {
        "vendorName": vendor,
        "invoiceNumber": invoice_number,
        "amountMinor": (
            amount_minor
            if isinstance(amount_minor, int)
            else 0
        ),
        "currency": currency or "INR"
    }

    rationale = (
        f"Proposed action: {action}. "
        + " ".join(reasons)
        + (
            " Supporting evidence references: "
            + ", ".join(decisive_refs)
            if decisive_refs
            else " No decisive source references were verified."
        )
    )[:1500]

    if len(rationale) < 60:
        rationale = rationale.ljust(60, ".")

    return {
        "packageId": package_id,
        "actionId": action_id,
        "action": action,
        "facts": facts,
        "evidenceRefs": decisive_refs,
        "rationale": rationale
    }


def validate_batch(data):
    if not isinstance(data, dict):
        return None, "Batch data must be a JSON object."

    if (
        not isinstance(data.get("batchId"), str)
        or not data["batchId"]
    ):
        return None, "batchId is required."

    if (
        not isinstance(data.get("policyRevision"), str)
        or not data["policyRevision"]
    ):
        return None, "policyRevision is required."

    packages = data.get("packages")

    if not isinstance(packages, list) or not packages:
        return None, "packages must be a non-empty list."

    if any(not isinstance(p, dict) for p in packages):
        return None, "Every package must be an object."

    ids = [str(p.get("packageId", "")) for p in packages]

    if any(not value for value in ids):
        return None, "Package IDs must be present and unique."

    if len(ids) != len(set(ids)):
        return None, "Package IDs must be present and unique."

    return packages, None


# ---------------------------------------------------------
# A2A message send
# ---------------------------------------------------------

@app.post("/a2a/message:send")
def message_send():
    header_error = check_protocol_headers()

    if header_error:
        return header_error

    owner, auth_error = require_owner()

    if auth_error:
        return auth_error

    body = request.get_json(silent=True)

    if (
        not isinstance(body, dict)
        or not isinstance(body.get("message"), dict)
    ):
        return error(
            "Expected an A2A request containing message."
        )

    message = body["message"]
    message_id = message.get("messageId")
    role = message.get("role")

    if not isinstance(message_id, str) or not message_id:
        return error("message.messageId is required.")

    if role != "ROLE_USER":
        return error("message.role must be ROLE_USER.")

    semantic_hash = canonical_hash(message)
    parts = get_message_parts(message)

    if not parts:
        return error("message.parts must not be empty.")

    with DB_LOCK:
        conn = connect_db()

        existing = conn.execute(
            """
            SELECT semantic_hash, task_id
            FROM messages
            WHERE owner=? AND message_id=?
            """,
            (owner, message_id)
        ).fetchone()

        if existing:
            if existing["semantic_hash"] != semantic_hash:
                conn.close()
                return error(
                    "Message ID reused with different content.",
                    409,
                    "IDEMPOTENCY_CONFLICT"
                )

            loaded = load_task(
                conn,
                existing["task_id"],
                owner
            )
            conn.close()

            if loaded:
                return json_response(task_envelope(loaded[1]))

            return error("Stored task is unavailable.", 500)

        batch_part = next(
            (
                p for p in parts
                if p.get("mediaType") == INPUT_MODE
            ),
            None
        )

        results_part = next(
            (
                p for p in parts
                if p.get("mediaType") == RESULT_MODE
            ),
            None
        )

        # Initial invoice batch.
        if batch_part is not None:
            data = batch_part.get("data")

            packages, validation_error = validate_batch(data)

            if validation_error:
                conn.close()
                return error(validation_error)

            task_id = str(uuid.uuid4())
            context_id = str(uuid.uuid4())

            proposals = []
            seen_actions = set()

            for package in packages:
                proposal = decide_package(
                    package,
                    data.get("policyRevision")
                )

                if proposal["actionId"] in seen_actions:
                    proposal["actionId"] = str(uuid.uuid4())

                seen_actions.add(proposal["actionId"])
                proposals.append(proposal)

            task = {
                "id": task_id,
                "contextId": context_id,
                "status": {
                    "state": "TASK_STATE_INPUT_REQUIRED"
                },
                "history": [message],
                "artifacts": [{
                    "artifactId": str(uuid.uuid4()),
                    "parts": [{
                        "mediaType": PROPOSAL_MODE,
                        "data": {
                            "batchId": data["batchId"],
                            "proposals": proposals
                        }
                    }]
                }]
            }

            save_task(
                conn,
                task_id,
                context_id,
                owner,
                "TASK_STATE_INPUT_REQUIRED",
                data["batchId"],
                task
            )

            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?)",
                (
                    owner,
                    message_id,
                    semantic_hash,
                    task_id
                )
            )

            for proposal in proposals:
                conn.execute(
                    """
                    INSERT INTO decisions
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        proposal["packageId"],
                        proposal["actionId"],
                        json.dumps(
                            proposal,
                            separators=(",", ":")
                        )
                    )
                )

            conn.commit()
            conn.close()

            return json_response(task_envelope(task))

        # Continuation containing the grader's results.
        if results_part is not None:
            task_id = message.get("taskId")
            context_id = message.get("contextId")
            data = results_part.get("data")

            if (
                not isinstance(task_id, str)
                or not isinstance(context_id, str)
            ):
                conn.close()
                return error(
                    "Result continuation needs taskId and contextId."
                )

            loaded = load_task(conn, task_id, owner)

            if not loaded:
                conn.close()
                return error(
                    "Task not found.",
                    404,
                    "TASK_NOT_FOUND"
                )

            row, task = loaded

            if row["context_id"] != context_id:
                conn.close()
                return error(
                    "Task context does not match.",
                    409,
                    "CONTEXT_MISMATCH"
                )

            if row["status"] != "TASK_STATE_INPUT_REQUIRED":
                conn.close()
                return error(
                    "Task is not awaiting results.",
                    409,
                    "INVALID_TASK_STATE"
                )

            if (
                not isinstance(data, dict)
                or data.get("batchId") != row["batch_id"]
            ):
                conn.close()
                return error(
                    "Result batch does not match the task.",
                    409,
                    "BATCH_MISMATCH"
                )

            results = data.get("results")

            if not isinstance(results, list):
                conn.close()
                return error("results must be a list.")

            proposal_data = find_artifact_data(
                task,
                PROPOSAL_MODE
            )

            if not isinstance(proposal_data, dict):
                conn.close()
                return error(
                    "Stored proposal artifact is missing.",
                    500
                )

            proposal_by_key = {
                (
                    p["packageId"],
                    p["actionId"],
                    p["action"]
                ): p
                for p in proposal_data.get("proposals", [])
            }

            accepted = []
            seen = set()

            for result in results:
                if not isinstance(result, dict):
                    continue

                key = (
                    result.get("packageId"),
                    result.get("actionId"),
                    result.get("action")
                )

                proposal = proposal_by_key.get(key)

                if not proposal or key in seen:
                    continue

                seen.add(key)

                if result.get("outcome") != "ACCEPTED":
                    continue

                nonce = result.get("receiptNonce")

                if not isinstance(nonce, str) or not nonce:
                    continue

                accepted_item = {
                    "packageId": proposal["packageId"],
                    "actionId": proposal["actionId"],
                    "action": proposal["action"],
                    "receiptNonce": nonce,
                    "facts": proposal["facts"],
                    "evidenceRefs": proposal["evidenceRefs"]
                }

                accepted.append(accepted_item)

                conn.execute(
                    """
                    INSERT OR IGNORE INTO receipts
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        proposal["packageId"],
                        proposal["actionId"],
                        nonce,
                        json.dumps(
                            accepted_item,
                            separators=(",", ":")
                        )
                    )
                )

            task["history"].append(message)
            task["status"] = {
                "state": "TASK_STATE_COMPLETED"
            }

            task["artifacts"].append({
                "artifactId": str(uuid.uuid4()),
                "parts": [{
                    "mediaType": RECEIPT_MODE,
                    "data": {
                        "batchId": row["batch_id"],
                        "executions": accepted
                    }
                }]
            })

            save_task(
                conn,
                task_id,
                context_id,
                owner,
                "TASK_STATE_COMPLETED",
                row["batch_id"],
                task
            )

            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?)",
                (
                    owner,
                    message_id,
                    semantic_hash,
                    task_id
                )
            )

            conn.commit()
            conn.close()

            return json_response(task_envelope(task))

        conn.close()

        return error(
            "Message must contain an invoice batch or result continuation."
        )


# ---------------------------------------------------------
# Task retrieval and listing
# ---------------------------------------------------------

@app.get("/a2a/tasks/<task_id>")
def get_task(task_id):
    header_error = check_protocol_headers()

    if header_error:
        return header_error

    owner, auth_error = require_owner()

    if auth_error:
        return auth_error

    with DB_LOCK:
        conn = connect_db()
        loaded = load_task(conn, task_id, owner)
        conn.close()

    if not loaded:
        return error(
            "Task not found.",
            404,
            "TASK_NOT_FOUND"
        )

    return json_response(task_envelope(loaded[1]))


@app.get("/a2a/tasks")
def list_tasks():
    header_error = check_protocol_headers()

    if header_error:
        return header_error

    owner, auth_error = require_owner()

    if auth_error:
        return auth_error

    with DB_LOCK:
        conn = connect_db()

        rows = conn.execute(
            """
            SELECT task_json
            FROM tasks
            WHERE owner=?
            ORDER BY created_at DESC
            """,
            (owner,)
        ).fetchall()

        conn.close()

    return json_response({
        "tasks": [
            json.loads(row["task_json"])
            for row in rows
        ]
    })


# ---------------------------------------------------------
# Task cancellation
# ---------------------------------------------------------

@app.post("/a2a/tasks/<task_id>:cancel")
def cancel_task(task_id):
    header_error = check_protocol_headers()

    if header_error:
        return header_error

    owner, auth_error = require_owner()

    if auth_error:
        return auth_error

    with DB_LOCK:
        conn = connect_db()
        loaded = load_task(conn, task_id, owner)

        if not loaded:
            conn.close()
            return error(
                "Task not found.",
                404,
                "TASK_NOT_FOUND"
            )

        row, task = loaded

        if row["status"] in (
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED"
        ):
            conn.close()
            return error(
                "Task is already terminal.",
                409,
                "TASK_TERMINAL"
            )

        task["status"] = {
            "state": "TASK_STATE_CANCELED"
        }

        save_task(
            conn,
            task_id,
            row["context_id"],
            owner,
            "TASK_STATE_CANCELED",
            row["batch_id"],
            task
        )

        conn.commit()
        conn.close()

    return json_response(task_envelope(task))


# ---------------------------------------------------------
# Error handling
# ---------------------------------------------------------

@app.errorhandler(404)
def not_found(_):
    return error(
        "Route not found.",
        404,
        "NOT_FOUND"
    )


@app.errorhandler(405)
def method_not_allowed(_):
    return error(
        "Method not allowed.",
        405,
        "METHOD_NOT_ALLOWED"
    )


@app.errorhandler(500)
def internal_error(_):
    return error(
        "Internal server error.",
        500,
        "INTERNAL_ERROR"
    )


# ---------------------------------------------------------
# Run locally / Render
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000))
    )
```
