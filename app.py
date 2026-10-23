import os
import json
import uuid
import hashlib
import hmac
import sqlite3
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from flask import Flask, jsonify, request, make_response

app = Flask(__name__)

AGENT_NAME = "GA3 Invoice Agent"
AGENT_VERSION = "1.0.0"
PUBLIC_URL = os.environ.get(
    "PUBLIC_URL", "https://tds-ga3-q2-invoice-agent.onrender.com"
).rstrip("/")
API_TOKEN = os.environ.get("API_TOKEN", "")
DATABASE = os.environ.get("DATABASE_PATH", "/tmp/ga3_tasks.db")

# Optional OpenAI-compatible endpoint. Configure these Render environment variables
# to use a hosted model; provider/model name itself is not graded.
AI_API_URL = os.environ.get("AI_API_URL", "").strip()
AI_API_KEY = os.environ.get("AI_API_KEY", "").strip()
AI_MODEL = os.environ.get("AI_MODEL", "").strip()

INPUT_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MODE = "application/vnd.ga5.invoice-action-receipts+json"
RESULT_MODE = "application/vnd.ga5.invoice-action-results+json"
A2A_JSON = "application/a2a+json"
A2A_VERSION = "1.0"
ACTIONS = {
    "settle_invoice", "request_approval", "hold_invoice",
    "reject_duplicate", "open_exception"
}
LOCK = threading.RLock()


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def response(data, status=200, media_type=A2A_JSON):
    res = make_response(jsonify(data), status)
    res.headers["Content-Type"] = f"{media_type}; charset=utf-8"
    res.headers["A2A-Version"] = A2A_VERSION
    return res


def error(code, message, status=400):
    return response({"error": {"code": code, "message": message}}, status)


def check_protocol():
    if request.headers.get("A2A-Version") != A2A_VERSION:
        return error("VERSION_MISMATCH", "Require A2A-Version: 1.0", 400)
    media = request.headers.get("Content-Type", "").split(";", 1)[0].strip()
    if media != A2A_JSON:
        return error("UNSUPPORTED_MEDIA_TYPE", "Require application/a2a+json", 415)
    return None

def authenticate():
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")

    if scheme.lower() != "bearer" or not token.strip():
        return None, error(
            "UNAUTHENTICATED",
            "Bearer authentication required",
            401
        )

    # Treat each exact Bearer token as a separate user.
    principal = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return principal, None

def connect():
    conn = sqlite3.connect(DATABASE, timeout=25, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=25000")
    return conn


def init_db():
    with LOCK:
        conn = connect()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            context_id TEXT NOT NULL,
            owner TEXT NOT NULL,
            status TEXT NOT NULL,
            batch_id TEXT NOT NULL,
            task_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            owner TEXT NOT NULL,
            message_id TEXT NOT NULL,
            semantic_hash TEXT NOT NULL,
            task_id TEXT NOT NULL,
            PRIMARY KEY(owner, message_id)
        );
        CREATE TABLE IF NOT EXISTS decisions (
            task_id TEXT NOT NULL,
            package_id TEXT NOT NULL,
            action_id TEXT NOT NULL,
            proposal_json TEXT NOT NULL,
            PRIMARY KEY(task_id, package_id),
            UNIQUE(task_id, action_id)
        );
        CREATE TABLE IF NOT EXISTS receipts (
            task_id TEXT NOT NULL,
            package_id TEXT NOT NULL,
            action_id TEXT NOT NULL,
            receipt_nonce TEXT NOT NULL,
            receipt_json TEXT NOT NULL,
            PRIMARY KEY(task_id, package_id),
            UNIQUE(task_id, action_id)
        );
        """)
        conn.close()


init_db()


def task_state(task):
    return task.get("status", {}).get("state", "")


def find_part(message, media_type):
    for part in message.get("parts", []) if isinstance(message.get("parts"), list) else []:
        if isinstance(part, dict) and part.get("mediaType") == media_type:
            return part
    return None


def artifact_data(task, media_type):
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("mediaType") == media_type:
                return part.get("data")
    return None


def load_task(conn, task_id, owner):
    row = conn.execute(
        "SELECT * FROM tasks WHERE task_id=? AND owner=?", (task_id, owner)
    ).fetchone()
    return (row, json.loads(row["task_json"])) if row else (None, None)


def store_task(conn, task_id, context_id, owner, status, batch_id, task):
    timestamp = now()
    conn.execute("""
        INSERT INTO tasks(task_id,context_id,owner,status,batch_id,task_json,created_at,updated_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(task_id) DO UPDATE SET
          status=excluded.status, task_json=excluded.task_json, updated_at=excluded.updated_at
    """, (task_id, context_id, owner, status, batch_id, canonical_json(task), timestamp, timestamp))


def get_text_values(obj):
    values = []
    def walk(value, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, str(k))
        elif isinstance(value, list):
            for item in value:
                walk(item, key)
        elif isinstance(value, (str, int, float, bool)):
            values.append((key, str(value)))
    walk(obj)
    return values


def call_ai(batch):
    """One batched model call. Returns packageId -> decision JSON."""
    if not (AI_API_URL and AI_MODEL):
        return None
    prompt = """
You are an evidence-based invoice decision agent. Read each package and its source
documents carefully; policy, negation, dates, payment records, and document context
matter. Ignore cover sheets, archive/training examples, irrelevant action words, and
old examples. For EACH package return exactly one action:
settle_invoice = valid, reconciled, and within autonomous authority;
request_approval = commercially valid but outside delegated authority;
hold_invoice = payment pauses pending a stated verification;
reject_duplicate = the same commercial invoice was already paid;
open_exception = material records conflict and require exception workflow.
Do not settle unless evidence proves validity, reconciliation, and authority. A payment
for a different invoice is not a duplicate. Never infer missing facts. Evidence refs
must be exactly the three decisive bracketed references from the paragraph that
determines the action; do not use cover-sheet, archive, or training-decoy references.
If three exact decisive references cannot be established, use open_exception and cite
only actual decisive refs available (do not fabricate). Return ONLY JSON:
{"proposals":[{"packageId":"exact ID","action":"one action","vendorName":"...","invoiceNumber":"...","amountMinor":123,"currency":"INR","evidenceRefs":["[ref]","[ref]","[ref]"],"rationale":"60-1500 characters; name action and cite at least two evidence refs"}]}
amountMinor is integer minor units. Preserve exact IDs and currency. Do not add fields.
""".strip()
    payload = {
        "model": AI_MODEL,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": canonical_json(batch)}
        ],
        "response_format": {"type": "json_object"}
    }
    headers = {"Content-Type": "application/json"}
    if AI_API_KEY:
        headers["Authorization"] = "Bearer " + AI_API_KEY
    req = urllib.request.Request(AI_API_URL, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=32) as res:
            result = json.loads(res.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        items = parsed.get("proposals")
        if not isinstance(items, list):
            return None
        return {str(x.get("packageId")): x for x in items if isinstance(x, dict)}
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError):
        return None


def fallback_decision(package):
    """Conservative fallback; AI is strongly recommended for the assignment's document-heavy cases."""
    text = "\n".join(v for _, v in get_text_values(package)).lower()
    vendor = str(package.get("vendorName", ""))
    invoice = str(package.get("invoiceNumber", ""))
    amount = package.get("amountMinor", 0)
    currency = str(package.get("currency", "INR"))
    refs = package.get("evidenceRefs", [])
    if not isinstance(refs, list):
        refs = []
    # Extract only literal bracketed citations found in source text.
    import re
    found = []
    for _, val in get_text_values(package):
        found.extend(re.findall(r"\[[^\[\]]+\]", val))
    found = list(dict.fromkeys(found))
    if not found:
        found = [r for r in refs if isinstance(r, str) and r.startswith("[") and r.endswith("]")]
    duplicate = any(x in text for x in ("same invoice already paid", "already paid this invoice",
                    "duplicate invoice paid", "payment for this invoice completed"))
    conflict = any(x in text for x in ("records conflict", "amount mismatch", "vendor mismatch",
                    "invoice number mismatch", "conflicting amount"))
    pending = any(x in text for x in ("verification pending", "awaiting verification",
                    "not verified", "verification required"))
    approval = any(x in text for x in ("approval required", "manager approval",
                    "outside delegated authority", "approval limit exceeded"))
    if duplicate:
        action, reason = "reject_duplicate", "Source text indicates this same commercial invoice was already paid."
    elif conflict:
        action, reason = "open_exception", "Source records contain a material conflict requiring exception review."
    elif pending:
        action, reason = "hold_invoice", "A stated verification requirement remains unresolved."
    elif approval:
        action, reason = "request_approval", "Source text indicates approval is required."
    else:
        action, reason = "open_exception", "Insufficient verified evidence to establish safe settlement authority."
    return {
        "packageId": str(package.get("packageId")),
        "action": action,
        "vendorName": vendor,
        "invoiceNumber": invoice,
        "amountMinor": amount if isinstance(amount, int) and not isinstance(amount, bool) else 0,
        "currency": currency,
        "evidenceRefs": found[:3],
        "rationale": (f"Proposed action: {action}. {reason} " +
                      ("Decisive source references: " + ", ".join(found[:3]) + "." if found
                       else "No decisive bracketed source references could be verified."))[:1500]
    }


def validate_batch(data):
    if not isinstance(data, dict):
        return "Batch data must be an object."
    if not isinstance(data.get("batchId"), str) or not data["batchId"]:
        return "batchId is required."
    if not isinstance(data.get("policyRevision"), str) or not data["policyRevision"]:
        return "policyRevision is required."
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        return "packages must be a non-empty list."
    ids = [p.get("packageId") for p in packages if isinstance(p, dict)]
    if len(ids) != len(packages) or any(not isinstance(x, str) or not x for x in ids):
        return "Every package needs a string packageId."
    if len(set(ids)) != len(ids):
        return "Package IDs must be unique."
    return None


def build_proposals(batch):
    ai = call_ai(batch)
    proposals = []
    used_ids = set()
    for package in batch["packages"]:
        package_id = package["packageId"]
        candidate = (ai or {}).get(package_id)
        if not isinstance(candidate, dict):
            candidate = fallback_decision(package)
        action = candidate.get("action")
        if action not in ACTIONS:
            candidate = fallback_decision(package)
            action = candidate["action"]
        refs = candidate.get("evidenceRefs", [])
        if not isinstance(refs, list):
            refs = []
        refs = [x for x in refs if isinstance(x, str) and x.startswith("[") and x.endswith("]")]
        # Keep exactly three references when the model found three; never fabricate missing refs.
        refs = list(dict.fromkeys(refs))[:3]
        facts = {
            "vendorName": str(candidate.get("vendorName", package.get("vendorName", "")))[:300],
            "invoiceNumber": str(candidate.get("invoiceNumber", package.get("invoiceNumber", "")))[:300],
            "amountMinor": candidate.get("amountMinor", package.get("amountMinor", 0)),
            "currency": str(candidate.get("currency", package.get("currency", "INR")))[:10]
        }
        if not isinstance(facts["amountMinor"], int) or isinstance(facts["amountMinor"], bool):
            facts["amountMinor"] = 0
        rationale = str(candidate.get("rationale", "")).strip()
        if len(rationale) < 60:
            rationale = (f"Proposed action: {action}. Evidence is insufficient to justify a more specific action. " +
                         ("References: " + ", ".join(refs) if refs else "No decisive references found."))[:1500]
        rationale = rationale[:1500]
        action_id = uuid.uuid4().hex
        while action_id in used_ids:
            action_id = uuid.uuid4().hex
        used_ids.add(action_id)
        proposals.append({
            "packageId": package_id,
            "actionId": action_id,
            "action": action,
            "facts": facts,
            "evidenceRefs": refs,
            "rationale": rationale
        })
    return {"batchId": batch["batchId"], "proposals": proposals}


@app.get("/")
def health():
    return jsonify({"service": AGENT_NAME, "status": "ok"})


@app.get("/.well-known/agent-card.json")
def agent_card():
    base = PUBLIC_URL + "/a2a/"
    return jsonify({
        "name": AGENT_NAME,
        "description": "Evidence-based invoice processing agent",
        "version": AGENT_VERSION,
        "url": base,
        "capabilities": {},
        "defaultInputModes": [INPUT_MODE],
        "defaultOutputModes": [PROPOSAL_MODE, RECEIPT_MODE],
        "supportedInterfaces": [{"protocolBinding": "HTTP+JSON",
                                 "protocolVersion": A2A_VERSION, "url": base}],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice Action Agent",
                    "description": "Reviews invoice evidence and proposes safe actions",
                    "tags": ["invoice", "payments", "evidence", "approval"]}]
    })


@app.post("/a2a/message:send")
def message_send():
    pe = check_protocol()
    if pe:
        return pe
    owner, ae = authenticate()
    if ae:
        return ae
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        return error("INVALID_ENVELOPE", "A message object is required")
    message = body["message"]
    if message.get("role") != "ROLE_USER":
        return error("INVALID_MESSAGE", "Expected ROLE_USER")
    message_id = message.get("messageId")
    if not isinstance(message_id, str) or not message_id:
        return error("INVALID_MESSAGE", "messageId is required")
    parts = message.get("parts")
    if not isinstance(parts, list) or not parts:
        return error("INVALID_MESSAGE", "message.parts must be a non-empty list")
    semantic_hash = digest(message)

    with LOCK:
        conn = connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT semantic_hash,task_id FROM messages WHERE owner=? AND message_id=?",
                (owner, message_id)).fetchone()
            if existing:
                if existing["semantic_hash"] != semantic_hash:
                    conn.rollback()
                    return error("IDEMPOTENCY_CONFLICT",
                                 "Message ID reused with different content", 409)
                _, old_task = load_task(conn, existing["task_id"], owner)
                conn.commit()
                if old_task is None:
                    return error("TASK_NOT_FOUND", "Stored task unavailable", 404)
                return response({"task": old_task})

            batch_part = find_part(message, INPUT_MODE)
            result_part = find_part(message, RESULT_MODE)
            if batch_part is not None:
                batch = batch_part.get("data")
                issue = validate_batch(batch)
                if issue:
                    conn.rollback()
                    return error("INVALID_BATCH", issue)
                proposals_data = build_proposals(batch)
                task_id = uuid.uuid4().hex
                context_id = message.get("contextId")
                if not isinstance(context_id, str) or not context_id:
                    context_id = uuid.uuid4().hex
                task = {
                    "id": task_id,
                    "contextId": context_id,
                    "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now()},
                    "history": [message],
                    "artifacts": [{"artifactId": "proposal-" + task_id,
                                   "parts": [{"mediaType": PROPOSAL_MODE,
                                              "data": proposals_data}]}],
                    "metadata": {}
                }
                conn.execute("""
                    INSERT INTO tasks(task_id,context_id,owner,status,batch_id,task_json,created_at,updated_at)
                    VALUES(?,?,?,?,?,?,?,?)
                """, (task_id, context_id, owner, "TASK_STATE_INPUT_REQUIRED",
                      batch["batchId"], canonical_json(task), now(), now()))
                conn.execute("INSERT INTO messages VALUES(?,?,?,?)",
                             (owner, message_id, semantic_hash, task_id))
                for proposal in proposals_data["proposals"]:
                    conn.execute("INSERT INTO decisions VALUES(?,?,?,?)",
                                 (task_id, proposal["packageId"], proposal["actionId"],
                                  canonical_json(proposal)))
                conn.commit()
                return response({"task": task})

            if result_part is None:
                conn.rollback()
                return error("INVALID_MESSAGE", "Expected invoice batch or result continuation")
            task_id = message.get("taskId")
            context_id = message.get("contextId")
            if not isinstance(task_id, str) or not isinstance(context_id, str):
                conn.rollback()
                return error("INVALID_CONTINUATION", "taskId and contextId are required")
            row, task = load_task(conn, task_id, owner)
            if row is None:
                conn.rollback()
                return error("TASK_NOT_FOUND", "Task not found", 404)
            if row["context_id"] != context_id or task.get("contextId") != context_id:
                conn.rollback()
                return error("CONTEXT_MISMATCH", "Task context does not match", 409)
            if row["status"] != "TASK_STATE_INPUT_REQUIRED":
                conn.rollback()
                return error("INVALID_TASK_STATE", "Task is not awaiting results", 409)
            data = result_part.get("data")
            if not isinstance(data, dict) or data.get("batchId") != row["batch_id"]:
                conn.rollback()
                return error("BATCH_MISMATCH", "Result batch does not match", 409)
            results = data.get("results")
            if not isinstance(results, list):
                conn.rollback()
                return error("INVALID_RESULTS", "results must be a list")
            stored = artifact_data(task, PROPOSAL_MODE)
            if not isinstance(stored, dict):
                conn.rollback()
                return error("STORED_PROPOSALS_MISSING", "Stored proposals unavailable", 500)
            proposal_by_key = {
                (p.get("packageId"), p.get("actionId"), p.get("action")): p
                for p in stored.get("proposals", []) if isinstance(p, dict)
            }
            seen = set()
            executions = []
            for result in results:
                if not isinstance(result, dict):
                    continue
                key = (result.get("packageId"), result.get("actionId"), result.get("action"))
                proposal = proposal_by_key.get(key)
                if proposal is None or key in seen:
                    continue
                seen.add(key)
                if result.get("outcome") != "ACCEPTED":
                    continue
                nonce = result.get("receiptNonce")
                if not isinstance(nonce, str) or not nonce:
                    continue
                prior = conn.execute(
                    "SELECT receipt_json,receipt_nonce FROM receipts WHERE task_id=? AND package_id=?",
                    (task_id, proposal["packageId"])).fetchone()
                if prior:
                    # A package can only execute once; replay cannot repeat action.
                    continue
                execution = {
                    "packageId": proposal["packageId"],
                    "actionId": proposal["actionId"],
                    "action": proposal["action"],
                    "receiptNonce": nonce,
                    "facts": proposal["facts"],
                    "evidenceRefs": proposal["evidenceRefs"]
                }
                conn.execute("INSERT INTO receipts VALUES(?,?,?,?,?)",
                             (task_id, proposal["packageId"], proposal["actionId"],
                              nonce, canonical_json(execution)))
                executions.append(execution)
            task["history"].append(message)
            task["artifacts"].append({
                "artifactId": "receipt-" + uuid.uuid4().hex,
                "parts": [{"mediaType": RECEIPT_MODE,
                           "data": {"batchId": row["batch_id"], "executions": executions}}]
            })
            task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now()}
            store_task(conn, task_id, context_id, owner, "TASK_STATE_COMPLETED",
                       row["batch_id"], task)
            conn.execute("INSERT INTO messages VALUES(?,?,?,?)",
                         (owner, message_id, semantic_hash, task_id))
            conn.commit()
            return response({"task": task})
        except sqlite3.IntegrityError:
            conn.rollback()
            return error("IDEMPOTENCY_CONFLICT", "Concurrent conflicting request", 409)
        except Exception:
            conn.rollback()
            app.logger.exception("message_send failed")
            return error("INTERNAL_ERROR", "Request could not be completed", 500)
        finally:
            conn.close()


@app.get("/a2a/tasks")
def list_tasks():
    pe = check_protocol()
    if pe:
        return pe
    owner, ae = authenticate()
    if ae:
        return ae
    with LOCK:
        conn = connect()
        rows = conn.execute("SELECT task_json FROM tasks WHERE owner=? ORDER BY created_at DESC",
                            (owner,)).fetchall()
        conn.close()
    return response({"tasks": [json.loads(r["task_json"]) for r in rows]})


@app.get("/a2a/tasks/<task_id>")
def read_task(task_id):
    pe = check_protocol()
    if pe:
        return pe
    owner, ae = authenticate()
    if ae:
        return ae
    with LOCK:
        conn = connect()
        _, task = load_task(conn, task_id, owner)
        conn.close()
    if task is None:
        return error("TASK_NOT_FOUND", "Task not found", 404)
    return response({"task": task})


@app.post("/a2a/tasks/<task_id>:cancel")
def cancel_task(task_id):
    pe = check_protocol()
    if pe:
        return pe
    owner, ae = authenticate()
    if ae:
        return ae
    with LOCK:
        conn = connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row, task = load_task(conn, task_id, owner)
            if row is None:
                conn.rollback()
                return error("TASK_NOT_FOUND", "Task not found", 404)
            if row["status"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
                conn.rollback()
                return error("TASK_TERMINAL", "Task is already terminal", 409)
            task["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now()}
            store_task(conn, task_id, row["context_id"], owner, "TASK_STATE_CANCELED",
                       row["batch_id"], task)
            conn.commit()
            return response({"task": task})
        except Exception:
            conn.rollback()
            app.logger.exception("cancel_task failed")
            return error("INTERNAL_ERROR", "Request could not be completed", 500)
        finally:
            conn.close()


@app.errorhandler(404)
def route_not_found(_):
    return error("NOT_FOUND", "Route not found", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return error("METHOD_NOT_ALLOWED", "Method not allowed", 405)


@app.errorhandler(500)
def internal_error(_):
    return error("INTERNAL_ERROR", "Internal server error", 500)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
