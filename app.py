import os
import json
import base64
import re
import time
import requests
from flask import Flask, redirect, request, session, url_for, render_template, jsonify, Response, stream_with_context
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

load_dotenv()

import sqlite3
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

DB_PATH = os.getenv("DB_PATH", "mailmind.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT,
            picture TEXT,
            credentials_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS digests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT NOT NULL,
            digest_text TEXT NOT NULL,
            email_count INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_user_credentials(email, name, picture, creds_dict):
    """Upsert a user's OAuth credentials so the background scheduler can use them later."""
    conn = get_db()
    conn.execute("""
        INSERT INTO users (email, name, picture, credentials_json, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            name=excluded.name,
            picture=excluded.picture,
            credentials_json=excluded.credentials_json,
            updated_at=excluded.updated_at
    """, (email, name, picture, json.dumps(creds_dict), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def update_user_token_only(email, creds_dict):
    """Refresh-only update — keeps name/picture untouched (used by the scheduler after a silent token refresh)."""
    conn = get_db()
    conn.execute(
        "UPDATE users SET credentials_json=?, updated_at=? WHERE email=?",
        (json.dumps(creds_dict), datetime.utcnow().isoformat(), email)
    )
    conn.commit()
    conn.close()

def save_digest(email, digest_text, email_count):
    conn = get_db()
    conn.execute(
        "INSERT INTO digests (user_email, digest_text, email_count, created_at) VALUES (?, ?, ?, ?)",
        (email, digest_text, email_count, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_latest_digest(email):
    conn = get_db()
    row = conn.execute(
        "SELECT digest_text, email_count, created_at FROM digests WHERE user_email=? ORDER BY created_at DESC LIMIT 1",
        (email,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

# Allow HTTP for local dev only — remove in production
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# Google sometimes returns the granted scopes in a different order (or with
# extra/reordered entries) than what was requested. oauthlib treats that as
# a hard error ("Scope has changed from ... to ...") by default, even though
# the actual permissions are identical — just reordered. This flag tells
# oauthlib to compare scopes as a set instead of an exact ordered match,
# which is what we want since we only care *which* scopes were granted.
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]

# Extended scopes that include send permission — used for the re-auth flow
# triggered by the Reply / Compose features. These are a superset of SCOPES
# so the existing read-only token remains valid after the user re-consents.
# gmail.send is sufficient to actually send mail via messages().send(); we
# also include gmail.compose since "Compose" is conceptually a create-draft-
# and-send action and some Google verification flows expect it alongside send.
SCOPES_WITH_SEND = SCOPES + [
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
]

# Extended scopes that additionally include Google Calendar event creation —
# used by the "Create Meeting" feature so we can generate a real Google Meet
# link and have the event show up on the user's actual Google Calendar, not
# just inside MailMind. This is a superset of SCOPES_WITH_SEND (which itself
# is a superset of SCOPES), so re-consenting here keeps Gmail send/read
# access intact too.
SCOPES_WITH_CALENDAR = SCOPES_WITH_SEND + [
    "https://www.googleapis.com/auth/calendar.events",
    # contacts.readonly lets the agent resolve a name ("Rahul", "HR") to a
    # real email address via the People API instead of grepping Gmail
    # headers with regex. Bundled into the same scope tier as calendar
    # since both are already behind one re-consent flow.
    "https://www.googleapis.com/auth/contacts.readonly",
]

CLIENT_SECRETS_FILE = "credentials.json"

# ==========================================
#   LLM PROVIDERS — primary (Groq) + automatic
#   fallback (OpenRouter → z-ai/glm-5.2) when
#   the primary hits a rate limit / token-budget
#   wall. Both are OpenAI-compatible chat
#   completions endpoints, so the same request
#   shape works for either — only url/key/model
#   change per provider.
#
#   NOTE: llama-3.3-70b-versatile and
#   llama-3.1-8b-instant are deprecated on Groq.
#   Default primary model is now openai/gpt-oss-120b
#   per Groq's current recommendation.
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "z-ai/glm-5.2")
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Ordered list of providers to try. "primary" always tried first; "fallback"
# only kicked in when primary raises GroqRateLimitError/GroqPayloadTooLargeError
# (i.e. token/rate budget exhausted) — not on genuine content/auth errors,
# so we don't silently swap models when the real problem is something else.
LLM_PROVIDERS = {
    "primary": {
        "name": "groq",
        "url": GROQ_CHAT_URL,
        "key": GROQ_API_KEY,
        "model": GROQ_MODEL,
    },
    "fallback": {
        "name": "nvidia",
        "url": NVIDIA_CHAT_URL,
        "key": NVIDIA_API_KEY,
        "model": NVIDIA_MODEL,
    },
}


# ==========================================
#              HELPERS
# ==========================================

def get_gmail_service():
    """Build Gmail API service from session credentials."""
    if "credentials" not in session:
        return None
    creds = Credentials(**session["credentials"])
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        session["credentials"] = credentials_to_dict(creds)
    return build("gmail", "v1", credentials=creds)


def get_calendar_service():
    """Build Google Calendar API service from session credentials."""
    if "credentials" not in session:
        return None
    creds = Credentials(**session["credentials"])
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        session["credentials"] = credentials_to_dict(creds)
    return build("calendar", "v3", credentials=creds)


def has_send_scope():
    """Return True if the stored credentials include the gmail.send scope."""
    creds_dict = session.get("credentials", {})
    granted = creds_dict.get("scopes") or []
    return "https://www.googleapis.com/auth/gmail.send" in granted


def has_calendar_scope():
    """Return True if the stored credentials include the calendar.events scope."""
    creds_dict = session.get("credentials", {})
    granted = creds_dict.get("scopes") or []
    return "https://www.googleapis.com/auth/calendar.events" in granted


def has_contacts_scope():
    """Return True if the stored credentials include the contacts.readonly scope."""
    creds_dict = session.get("credentials", {})
    granted = creds_dict.get("scopes") or []
    return "https://www.googleapis.com/auth/contacts.readonly" in granted


def get_people_service():
    """Build the Google People API service from session credentials."""
    if "credentials" not in session:
        return None
    creds = Credentials(**session["credentials"])
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        session["credentials"] = credentials_to_dict(creds)
    return build("people", "v1", credentials=creds)


def credentials_to_dict(creds):
    return {
        "token":         creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri":     creds.token_uri,
        "client_id":     creds.client_id,
        "client_secret": creds.client_secret,
        "scopes":        creds.scopes,
    }


def fetch_messages_batch(service, message_ids, fmt="full", metadata_headers=None):
    """
    Fetch many Gmail messages in parallel using the batch API instead of
    one sequential .get() call per message. This turns N sequential round
    trips into a small number of batched HTTP requests (Gmail batches in
    groups of up to 100), which is the main reason the AI agent endpoints
    used to feel slow (~10-20s for 40-50 emails).

    fmt="metadata" (with metadata_headers, e.g. ["From","To","Subject","Date"])
    skips downloading the full MIME body entirely — much smaller payload and
    faster parsing than fmt="full". Use this for any caller that only reads
    headers/snippet/labelIds (build_email_obj with include_body=False), and
    reserve fmt="full" for callers that actually call extract_body().
    """
    results = {}
    errors = []

    def _callback(request_id, response, exception):
        if exception is not None:
            errors.append((request_id, str(exception)))
        else:
            results[request_id] = response

    # Gmail API allows up to 100 calls per batch request.
    for i in range(0, len(message_ids), 100):
        chunk = message_ids[i:i + 100]
        batch = service.new_batch_http_request(callback=_callback)
        for msg_id in chunk:
            kwargs = {"userId": "me", "id": msg_id, "format": fmt}
            if fmt == "metadata" and metadata_headers:
                kwargs["metadataHeaders"] = metadata_headers
            batch.add(
                service.users().messages().get(**kwargs),
                request_id=msg_id,
            )
        batch.execute()

    return results, errors


# Headers build_email_obj actually reads — pass this to fetch_messages_batch
# whenever fmt="metadata" is used, so Gmail doesn't have to include every
# header (some emails carry 30+ headers) in the response.
EMAIL_METADATA_HEADERS = ["From", "To", "Subject", "Date"]


def build_email_obj(msg_id, detail, include_body=False, body_char_limit=2000):
    """Shared helper to turn a raw Gmail message detail into our email dict."""
    payload = detail.get("payload", {})
    headers = {h["name"]: h["value"] for h in payload.get("headers", [])}
    label_ids = detail.get("labelIds", [])
    obj = {
        "id":        msg_id,
        "from":      headers.get("From", "Unknown Sender"),
        "to":        headers.get("To", ""),
        "subject":   headers.get("Subject", "(no subject)"),
        "date":      headers.get("Date", ""),
        "snippet":   detail.get("snippet", ""),
        "unread":    "UNREAD" in label_ids,
        "important": "IMPORTANT" in label_ids,
    }
    if include_body:
        obj["body"] = extract_body(payload)[:body_char_limit]
    return obj


# How many emails we'll send to Groq in a single agent call.
# llama-3.1-8b-instant free tier: ~30K TPM but per-request TPM is tight.
# Keeping this at 10 ensures we stay well under the per-request limit.
MAX_EMAILS_FOR_AI = 10

# How many recent emails to pull from Gmail as the candidate pool before
# picking MAX_EMAILS_FOR_AI of them for the AI call. Was hardcoded to 40 —
# dropped to 20 (still 2x the AI batch size) since fetching/parsing double
# what's actually scored was pure latency with no benefit. Raise this if
# you want a bigger candidate pool (e.g. for Categorize's fallback bucket
# to cover more of the inbox), at the cost of a slower Gmail round trip.
AGENT_EMAIL_FETCH_POOL = 20


def trim_field(text, limit=200):
    """Truncate a header/snippet field before sending it to the AI. Some
    senders (newsletters especially) put unusually long encoded text in
    From/Subject headers, which can otherwise bloat the request payload."""
    return (text or "")[:limit]


def extract_body(payload):
    """Recursively extract plain-text or HTML body from a Gmail message payload."""
    body_text = ""
    if "parts" in payload:
        for part in payload["parts"]:
            mime = part.get("mimeType", "")
            if mime == "text/plain":
                data = part.get("body", {}).get("data", "")
                if data:
                    body_text = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
                    return body_text
            elif mime == "text/html" and not body_text:
                data = part.get("body", {}).get("data", "")
                if data:
                    html = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
                    # Strip HTML tags for plain text
                    body_text = re.sub(r"<[^>]+>", " ", html)
                    body_text = re.sub(r"\s+", " ", body_text).strip()
            elif "parts" in part:
                nested = extract_body(part)
                if nested:
                    return nested
    else:
        data = payload.get("body", {}).get("data", "")
        if data:
            raw = base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
            mime = payload.get("mimeType", "")
            if mime == "text/html":
                raw = re.sub(r"<[^>]+>", " ", raw)
                raw = re.sub(r"\s+", " ", raw).strip()
            body_text = raw
    return body_text


class GroqRateLimitError(Exception):
    """Raised when the active provider returns 429 Too Many Requests (after
    exhausting retries), with a friendlier message."""
    pass


class GroqPayloadTooLargeError(Exception):
    """Raised when the active provider returns 413 Payload Too Large."""
    pass


class AllProvidersExhaustedError(Exception):
    """Raised when both the primary and fallback providers failed with a
    rate-limit/budget error — nothing left to fall back to."""
    pass


def _llm_call_raw(provider, system_prompt, user_content, stream=False, max_retries=4, max_tokens=4096):
    """
    Make a single chat-completion call against one specific provider config
    (a dict with url/key/model, from LLM_PROVIDERS). Returns response object
    if stream=True, else parsed JSON. On 429, retries with backoff honouring
    Retry-After if present. Raises GroqPayloadTooLargeError/GroqRateLimitError
    on unrecoverable 413/429 so the caller (llm_call) can decide whether to
    fail outright or fall back to the next provider.
    """
    if not provider["key"]:
        raise ValueError(f"No API key configured for provider '{provider['name']}'.")

    payload = {
        "model": provider["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        "stream": stream,
        "max_tokens": max_tokens,
    }

    payload_bytes = len(json.dumps(payload).encode("utf-8"))
    print(f"[llm_call:{provider['name']}] sending payload: {payload_bytes} bytes, "
          f"model={provider['model']}, system={len(system_prompt)} chars, "
          f"user={len(user_content)} chars, max_tokens={max_tokens}")

    headers = {"Authorization": f"Bearer {provider['key']}", "Content-Type": "application/json"}
    # OpenRouter asks that requests identify the calling app — harmless to
    # omit but recommended by their docs; doesn't affect Groq requests since
    # this header is simply ignored there.
    if provider["name"] == "openrouter":
        headers["HTTP-Referer"] = "https://mailmind.local"
        headers["X-Title"] = "MailMind"

    attempt = 0
    while True:
        resp = requests.post(
            provider["url"],
            headers=headers,
            json=payload,
            timeout=60,
            stream=stream,
        )

        if resp.status_code == 413:
            # Both Groq and OpenRouter return 413 both for genuinely oversized
            # request bodies AND for exceeding a model's tokens-per-minute
            # (TPM) budget in a single request. Retrying immediately won't
            # help (TPM resets on a rolling window), so surface a clear error
            # rather than looping — the caller decides whether to fall back.
            raise GroqPayloadTooLargeError(
                f"{provider['name']}'s per-minute token limit (TPM) was hit in a single "
                f"request. The email batch has already been reduced to help."
            )

        if resp.status_code == 429:
            if attempt >= max_retries:
                raise GroqRateLimitError(
                    f"{provider['name']} rate limit reached (too many AI requests in a short time)."
                )
            retry_after = resp.headers.get("Retry-After")
            if retry_after:
                wait_s = float(retry_after)
            else:
                wait_s = min(2 ** (attempt + 1) + (attempt * 0.5), 30)
            print(f"[llm_call:{provider['name']}] 429 rate limit — waiting {wait_s:.1f}s before retry {attempt+1}/{max_retries}")
            time.sleep(wait_s)
            attempt += 1
            continue

        resp.raise_for_status()
        if stream:
            return resp
        return resp.json()


def llm_call(system_prompt, user_content, stream=False, max_retries=2, max_tokens=4096):
    """
    Provider-aware chat completion with automatic failover: tries the
    primary provider (Groq) first; if it exhausts retries on a rate-limit
    (429) or per-request token-budget (413) error, automatically retries
    once against the fallback provider (OpenRouter → z-ai/glm-5.2) with no
    extra action needed from the caller or the user — the switch is silent.

    Only rate/budget errors trigger failover. Auth errors, missing API
    keys, and genuine content problems are NOT retried on the fallback,
    since swapping providers won't fix those and would just mask the real
    issue.

    max_retries is per-provider (so a genuinely stuck primary doesn't burn
    through 2x the retries before failing over — lower than the old
    single-provider default since we now have a second provider as backup
    instead of retrying the same one forever).
    """
    try:
        return _llm_call_raw(LLM_PROVIDERS["primary"], system_prompt, user_content,
                              stream=stream, max_retries=max_retries, max_tokens=max_tokens)
    except (GroqRateLimitError, GroqPayloadTooLargeError) as primary_err:
        fallback = LLM_PROVIDERS["fallback"]
        if not fallback["key"]:
            # No fallback configured — surface the original error rather
            # than a confusing "no key" error for a provider the user never
            # set up.
            raise primary_err
        print(f"[llm_call] primary ({LLM_PROVIDERS['primary']['name']}) exhausted "
              f"({primary_err}) — switching to fallback ({fallback['name']}/{fallback['model']})")
        try:
            return _llm_call_raw(fallback, system_prompt, user_content,
                                  stream=stream, max_retries=max_retries, max_tokens=max_tokens)
        except (GroqRateLimitError, GroqPayloadTooLargeError) as fallback_err:
            raise AllProvidersExhaustedError(
                f"Both providers are currently rate-limited or over budget "
                f"(primary: {primary_err}; fallback: {fallback_err}). Please wait a minute and try again."
            )


# Backward-compatible alias — existing call sites throughout this file use
# groq_call(...); keep that name working while the implementation now does
# provider failover under the hood.
groq_call = llm_call


def groq_json_call(system_prompt, user_content, retries=1, max_tokens=1500):
    """
    Call Groq expecting a JSON response, with markdown-fence stripping and
    one automatic retry if the model returns invalid JSON. Returns the
    parsed Python object. Raises ValueError with the raw content if all
    attempts fail, so callers can return a clean error instead of a raw
    500 traceback.

    max_tokens defaults to 1500 (down from the old blanket 4096) — a JSON
    array for 10 emails' worth of scores/reasons needs a few hundred tokens,
    not 4096; a smaller ceiling reduces generation latency and lowers the
    chance of tripping the free-tier per-request TPM limit.
    """
    last_raw = ""
    for attempt in range(retries + 1):
        result = groq_call(system_prompt, user_content, max_tokens=max_tokens)
        content = result["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
        last_raw = content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Sometimes the model wraps the array in extra prose or adds a
            # trailing comma. Try to salvage just the [...] portion before
            # giving up and retrying.
            match = re.search(r"\[.*\]", content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            continue  # retry the whole call
    raise ValueError(f"AI returned invalid JSON after {retries + 1} attempt(s): {last_raw[:300]}")


# ==========================================
#              FRONTEND ROUTES
# ==========================================

@app.route("/")
def index():
    if "credentials" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html", logged_in=False, user_email=None)


@app.route("/dashboard")
def dashboard():
    if "credentials" not in session:
        return redirect(url_for("index"))
    return render_template(
        "dashboard.html",
        user_email=session.get("user_email"),
        user_name=session.get("user_name"),
        user_pic=session.get("user_pic"),
    )


# ==========================================
#             GOOGLE OAUTH ROUTES
# ==========================================

@app.route("/login")
def login():
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES)
    flow.redirect_uri = url_for("oauth_callback", _external=True)
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    session["state"] = state
    session["code_verifier"] = flow.code_verifier
    # Remember which scope set this flow was started with so the callback
    # can rebuild the Flow with the SAME scopes — using the wrong scope list
    # in the callback can itself contribute to invalid_grant/scope-change
    # errors during the token exchange.
    session["oauth_scopes"] = SCOPES
    return redirect(authorization_url)


@app.route("/login/with-send")
def login_with_send():
    """
    Re-consent flow that adds the gmail.send scope so the Reply feature works.
    The user is redirected here from the frontend when they click Reply for
    the first time without having the send scope yet.
    """
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES_WITH_SEND)
    flow.redirect_uri = url_for("oauth_callback", _external=True)
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"   # force re-consent so Google shows the new scope
    )
    session["state"] = state
    session["code_verifier"] = flow.code_verifier
    session["oauth_scopes"] = SCOPES_WITH_SEND
    return redirect(authorization_url)


@app.route("/login/with-calendar")
def login_with_calendar():
    """
    Re-consent flow that adds the calendar.events scope so the "Create
    Meeting" feature can create real Google Calendar events (with an
    auto-generated Google Meet link) and send the invite email. The user is
    redirected here from the frontend the first time they try to create a
    meeting without having the calendar scope yet.

    Also grants contacts.readonly (bundled into SCOPES_WITH_CALENDAR), which
    the agent's find_contact_email tool uses to resolve a name like "Rahul"
    or "HR" to a real address via the People API.
    """
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=SCOPES_WITH_CALENDAR)
    flow.redirect_uri = url_for("oauth_callback", _external=True)
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    session["state"] = state
    session["code_verifier"] = flow.code_verifier
    session["oauth_scopes"] = SCOPES_WITH_CALENDAR
    return redirect(authorization_url)


@app.route("/oauth/callback")
def oauth_callback():
    # ── Guard against the same authorization code being exchanged twice ──
    incoming_code = request.args.get("code")
    if incoming_code and session.get("last_oauth_code") == incoming_code and "credentials" in session:
        return redirect(url_for("dashboard"))

    state = session.get("state")
    scopes_used = session.get("oauth_scopes", SCOPES)
    flow = Flow.from_client_secrets_file(CLIENT_SECRETS_FILE, scopes=scopes_used, state=state)
    flow.redirect_uri = url_for("oauth_callback", _external=True)
    flow.code_verifier = session.get("code_verifier")

    try:
        flow.fetch_token(authorization_response=request.url)
    except Exception:
        if "credentials" in session:
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    if incoming_code:
        session["last_oauth_code"] = incoming_code

    credentials = flow.credentials
    session["credentials"] = credentials_to_dict(credentials)
    print(f"[oauth_callback] granted scopes: {credentials.scopes}")

    user_info_service = build("oauth2", "v2", credentials=credentials)
    user_info = user_info_service.userinfo().get().execute()
    session["user_email"] = user_info.get("email")
    session["user_name"]  = user_info.get("name")
    session["user_pic"]   = user_info.get("picture")

    # NEW — persist creds to the DB so the 7 PM scheduler can find this user
    save_user_credentials(
        session["user_email"], session["user_name"], session["user_pic"],
        session["credentials"]
    )

    return redirect(url_for("dashboard"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


# ==========================================
#              GMAIL APIs
# ==========================================

@app.route("/api/emails/recent")
def api_recent_emails():
    """
    Fetch the latest emails.
    - ?max=N  sets how many to fetch (default 30)
    - ?full=1  includes full decoded body
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    service = get_gmail_service()
    max_results = min(int(request.args.get("max", 30)), 100)
    include_body = request.args.get("full", "0") == "1"

    try:
        results = service.users().messages().list(userId="me", maxResults=max_results).execute()
        messages = results.get("messages", [])
        if not messages:
            return jsonify({"emails": [], "total": 0})

        ids = [m["id"] for m in messages]
        fetch_fmt = "full" if include_body else "metadata"
        details, errors = fetch_messages_batch(service, ids, fmt=fetch_fmt, metadata_headers=EMAIL_METADATA_HEADERS)

        emails = []
        for msg_id in ids:
            detail = details.get(msg_id)
            if not detail:
                continue  # skip ids that failed in the batch (logged in errors)
            emails.append(build_email_obj(msg_id, detail, include_body=include_body, body_char_limit=4000))

        return jsonify({"emails": emails, "total": len(emails)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/emails/<string:email_id>/body")
def api_email_body(email_id):
    """Fetch the full decoded body of a single email, and mark it as read
    in Gmail (since opening an email in a real inbox always marks it read)."""
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    service = get_gmail_service()
    try:
        detail = service.users().messages().get(userId="me", id=email_id, format="full").execute()
        payload = detail.get("payload", {})
        headers_list = payload.get("headers", [])
        headers = {h["name"]: h["value"] for h in headers_list}
        was_unread = "UNREAD" in detail.get("labelIds", [])

        body = extract_body(payload)

        # Mark as read in Gmail, same as opening a message in the real
        # Gmail UI would. Don't let this fail the whole request if it
        # errors for some reason (e.g. message already read) — the body
        # was already fetched successfully.
        if was_unread:
            try:
                service.users().messages().modify(
                    userId="me", id=email_id, body={"removeLabelIds": ["UNREAD"]}
                ).execute()
            except Exception:
                pass

        return jsonify({
            "id":      email_id,
            "subject": headers.get("Subject", "(no subject)"),
            "from":    headers.get("From", "Unknown"),
            "to":      headers.get("To", ""),
            "date":    headers.get("Date", ""),
            "body":    body,
            "snippet": detail.get("snippet", ""),
            "was_unread": was_unread,  # tells the frontend to decrement unread counters
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def api_stats():
    """Exact inbox-level counts for the dashboard cards."""
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    service = get_gmail_service()
    try:
        inbox     = service.users().labels().get(userId="me", id="INBOX").execute()
        important = service.users().labels().get(userId="me", id="IMPORTANT").execute()
        sent      = service.users().labels().get(userId="me", id="SENT").execute()
        return jsonify({
            "total":     inbox.get("messagesTotal", 0),
            "unread":    inbox.get("messagesUnread", 0),
            "important": important.get("messagesTotal", 0),
            "sent":      sent.get("messagesTotal", 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#              DIGESTS
# ==========================================

@app.route("/api/digest/latest")
def api_digest_latest():
    """Returns the most recent stored digest for this user (written by the
    7 PM scheduler), for the Overview card."""
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    digest = get_latest_digest(session.get("user_email"))
    if not digest:
        return jsonify({"digest": None})
    return jsonify({"digest": digest})


@app.route("/api/digest/history")
def api_digest_history():
    """Returns the last N stored digests for this user, newest first — for
    an AI Summarize 'Past Digests' list."""
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    limit = min(int(request.args.get("limit", 14)), 60)
    conn = get_db()
    rows = conn.execute(
        "SELECT id, digest_text, email_count, created_at FROM digests "
        "WHERE user_email=? ORDER BY created_at DESC LIMIT ?",
        (session.get("user_email"), limit)
    ).fetchall()
    conn.close()
    return jsonify({"digests": [dict(r) for r in rows]})


# ==========================================
#         AI AGENT: IMPORTANT EMAILS
# ==========================================

@app.route("/api/agent/important")
def api_agent_important():
    """
    AI Agent that reads recent emails and returns only the truly important ones.
    Uses Groq to score importance beyond Gmail's own IMPORTANT label,
    understanding context, sender reputation, urgency signals, etc.
    Returns JSON array with importance_reason for each email.
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    service = get_gmail_service()
    if not GROQ_API_KEY and not NVIDIA_API_KEY:
        return jsonify({"error": "No AI provider configured — set GROQ_API_KEY and/or NVIDIA_API_KEY"}), 500

    try:
        results = service.users().messages().list(userId="me", maxResults=AGENT_EMAIL_FETCH_POOL).execute()
        messages = results.get("messages", [])
        if not messages:
            return jsonify({"emails": []})

        ids = [m["id"] for m in messages]
        details, _errors = fetch_messages_batch(service, ids, fmt="metadata", metadata_headers=EMAIL_METADATA_HEADERS)

        emails = []
        for msg_id in ids:
            detail = details.get(msg_id)
            if not detail:
                continue
            base = build_email_obj(msg_id, detail)
            label_ids = detail.get("labelIds", [])
            base["gmail_imp"] = "IMPORTANT" in label_ids
            emails.append(base)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not emails:
        return jsonify({"emails": []})

    ai_emails = emails[:MAX_EMAILS_FOR_AI]

    email_list_text = "\n\n".join([
        f"[{i+1}] ID:{e['id']}\nFrom: {trim_field(e["from"], 60)}\nSubject: {trim_field(e["subject"], 80)}\nDate: {e['date']}\nSnippet: {trim_field(e['snippet'], 100)}"
        for i, e in enumerate(ai_emails)
    ])

    system_prompt = """You are an expert email importance classifier agent.
Your job is to analyse a list of emails and determine which ones are TRULY IMPORTANT for the user.

Classify as important if the email:
- Is from a real person (not automated/marketing) who needs a response
- Contains a meeting invite, interview confirmation, or job offer
- Has an action required (payment, verification, deadline, approval)
- Contains time-sensitive information (offer expiry, event today/tomorrow)
- Is from a known important sender (boss, client, bank, government)
- Contains security alerts or account issues
- Is flagged IMPORTANT by Gmail (gmail_imp: true)

Do NOT classify as important:
- Newsletters, promotions, marketing emails
- Automated notifications that require no action
- Social media notifications
- Spam or phishing

Respond ONLY with a valid JSON array (no markdown, no backticks, no extra text).
Each object in the array must have exactly these fields:
{
  "id": "<the exact email id from the input>",
  "importance_score": <integer 1-10>,
  "importance_reason": "<one sentence explaining why it's important>",
  "action_needed": "<what the user should do, or 'No action needed'>"
}

Only include emails with importance_score >= 6. Sort by score descending."""

    try:
        important_list = groq_json_call(system_prompt, email_list_text)

        # Merge AI results back with original email data
        id_map = {e["id"]: e for e in emails}
        enriched = []
        for item in important_list:
            base = id_map.get(item.get("id"), {})
            if base:
                enriched.append({**base, **item})

        return jsonify({"emails": enriched, "total": len(enriched), "source": "ai_agent"})

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#         AI AGENT: MEETINGS EXTRACTOR
# ==========================================

@app.route("/api/agent/meetings")
def api_agent_meetings():
    """
    AI Agent that reads recent emails and extracts meeting/event information
    including exact dates, times, links, and calendar data.
    Much more accurate than keyword search — uses LLM to understand context.
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    service = get_gmail_service()
    if not GROQ_API_KEY and not NVIDIA_API_KEY:
        return jsonify({"error": "No AI provider configured — set GROQ_API_KEY and/or NVIDIA_API_KEY"}), 500

    # Search for meeting-related emails (broader than before)
    query = "newer_than:30d (meeting OR interview OR call OR zoom OR meet OR webinar OR invite OR calendar OR schedule OR appointment OR standup OR sync OR demo OR presentation)"

    try:
        results = service.users().messages().list(userId="me", q=query, maxResults=20).execute()
        messages = results.get("messages", [])
        if not messages:
            return jsonify({"meetings": []})

        ids = [m["id"] for m in messages]
        details, _errors = fetch_messages_batch(service, ids, fmt="full")

        emails = []
        for msg_id in ids:
            detail = details.get(msg_id)
            if not detail:
                continue
            emails.append(build_email_obj(msg_id, detail, include_body=True, body_char_limit=2000))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not emails:
        return jsonify({"meetings": []})

    email_list_text = "\n\n".join([
        f"[{i+1}] ID:{e['id']}\nFrom: {trim_field(e["from"], 60)}\nSubject: {trim_field(e["subject"], 80)}\nEmail Date: {e['date']}\nBody preview: {trim_field(e.get('body', ''), 300)}"
        for i, e in enumerate(emails)
    ])

    system_prompt = """You are a calendar and meeting extraction AI agent.
Analyse the emails and extract structured meeting/event information.

For each email that contains a REAL meeting, call, interview, or event:
- Extract the exact date and time if mentioned
- Extract the meeting type (interview, standup, zoom call, webinar, etc.)
- Extract the meeting link if present (Zoom, Meet, Teams, etc.)
- Extract who organised it and who else is involved

Respond ONLY with a valid JSON array (no markdown, no backticks).
Each object must have:
{
  "id": "<exact email id>",
  "from": "<sender>",
  "subject": "<email subject>",
  "email_date": "<when the email was sent>",
  "meeting_type": "<interview|standup|zoom_call|webinar|client_call|team_meeting|other>",
  "meeting_date": "<extracted date in YYYY-MM-DD format, or null if not found>",
  "meeting_time": "<extracted time like 3:00 PM IST, or null>",
  "meeting_day": "<day name like Monday, or null>",
  "meeting_link": "<zoom/meet/teams URL or null>",
  "organizer": "<who organised it>",
  "description": "<one sentence summary of what this meeting is about>",
  "calendar_label": "<short label for calendar dot, max 20 chars>"
}

Only include emails where a real upcoming or recent meeting is clearly mentioned.
If no meetings found, return empty array []."""

    try:
        meetings = groq_json_call(system_prompt, email_list_text)
        return jsonify({"meetings": meetings, "total": len(meetings), "source": "ai_agent"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#         AI AGENT: CATEGORIZE
#   (priority + importance are folded into this
#   single call — see MailMind improvement plan,
#   item 3: "Remove the separate Priority section")
# ==========================================

def _fallback_priority_for(email):
    """
    Deterministic, non-AI fallback used only for emails the model didn't
    return a priority for (capped batch, or a skipped item). Keeps every
    email in a usable state instead of silently missing priority fields.
    """
    priority = "MEDIUM" if email.get("gmail_imp") else "LOW"
    score = 5 if email.get("gmail_imp") else 2
    return {
        "priority": priority,
        "priority_score": score,
        "priority_reason": "Not scored by AI (batch limit) — using Gmail's own IMPORTANT flag as a rough signal." if email.get("gmail_imp")
                            else "Not scored by AI (batch limit) — defaulted to low priority.",
    }


@app.route("/api/agent/categorize")
def api_agent_categorize():
    """
    AI Agent that categorises emails into Gmail-like tabs/folders (Primary,
    Social, Promotions, Updates, Finance, Jobs, Security, Newsletters) AND,
    in the same Groq call, assigns each email a priority level/score.

    Priority used to be its own agent + its own page (/api/agent/priority,
    the "Priority Inbox" view). Per the MailMind improvement plan, that's
    now folded in here: priority is just another attribute on a categorized
    email, not a separate thing the user has to run separately. This also
    halves the number of Groq round-trips needed to fully triage an inbox.
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    service = get_gmail_service()
    if not GROQ_API_KEY and not NVIDIA_API_KEY:
        return jsonify({"error": "No AI provider configured — set GROQ_API_KEY and/or NVIDIA_API_KEY"}), 500

    try:
        results = service.users().messages().list(userId="me", maxResults=AGENT_EMAIL_FETCH_POOL).execute()
        messages = results.get("messages", [])
        if not messages:
            return jsonify({"categories": {}})

        ids = [m["id"] for m in messages]
        details, _errors = fetch_messages_batch(service, ids, fmt="metadata", metadata_headers=EMAIL_METADATA_HEADERS)

        emails = []
        for msg_id in ids:
            detail = details.get(msg_id)
            if not detail:
                continue
            base = build_email_obj(msg_id, detail)
            base["gmail_imp"] = "IMPORTANT" in detail.get("labelIds", [])
            emails.append(base)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not emails:
        return jsonify({"categories": {}})

    ai_emails = emails[:MAX_EMAILS_FOR_AI]

    email_list_text = "\n\n".join([
        f"[{i+1}] ID:{e['id']}\nFrom: {trim_field(e["from"], 60)}\nSubject: {trim_field(e["subject"], 80)}\nSnippet: {trim_field(e['snippet'], 100)}\nGmail-flagged important: {e['gmail_imp']}"
        for i, e in enumerate(ai_emails)
    ])

    system_prompt = """You are an email triage AI agent — like Gmail's smart tabs, but smarter,
and it also tells the user what to look at first.

For EVERY email, do two things at once:

1) CATEGORISE into exactly one of these:
- Primary: Direct personal/professional emails from real people, direct replies
- Social: Facebook, Instagram, LinkedIn, Twitter/X, WhatsApp, dating apps, community
- Promotions: Sales, discounts, marketing, e-commerce, shopping, offers, coupons
- Updates: Automated notifications, receipts, shipping, app updates, GitHub, CI/CD
- Finance: Banks, payments, invoices, transactions, UPI, credit cards, taxes
- Jobs: Job boards, recruiters, HR, interview calls, placement cells, internships
- Security: OTPs, login alerts, password resets, 2FA, suspicious activity
- Newsletters: Subscriptions, digest emails, blog posts, weekly roundups

2) ASSIGN A PRIORITY:
- HIGH (score 8-10): Needs immediate attention. Job offers, interviews, payment due, security alerts, direct personal messages requiring reply, deadlines today/tomorrow.
- MEDIUM (score 5-7): Should be addressed soon. Meeting invites, task updates, follow-ups, professional correspondence, important notifications.
- LOW (score 1-4): Can wait. Newsletters, promotions, social notifications, automated reports, FYI updates.

You MUST include EVERY email.

Respond ONLY with a valid JSON array (no markdown, no backticks, no extra text).
Each object must have exactly:
{
  "id": "<exact email id>",
  "category": "<one of the 8 categories above>",
  "category_reason": "<3-5 word reason>",
  "priority": "<HIGH|MEDIUM|LOW>",
  "priority_score": <integer 1-10>,
  "priority_reason": "<one concise sentence why this priority>"
}"""

    try:
        cat_list = groq_json_call(system_prompt, email_list_text)

        id_map = {e["id"]: e for e in emails}

        # Build category → emails map, each email carrying its own
        # priority/priority_score/priority_reason alongside category info.
        categories = {}
        for item in cat_list:
            base = id_map.get(item.get("id"), {})
            if not base:
                continue
            cat = item.get("category", "Primary")
            enriched = {
                **base,
                "category_reason": item.get("category_reason", ""),
                "priority": item.get("priority", "LOW"),
                "priority_score": item.get("priority_score", 1),
                "priority_reason": item.get("priority_reason", ""),
            }
            categories.setdefault(cat, []).append(enriched)

        # Any email the AI didn't return a row for (batch cap, or a skipped
        # item) still needs to show up somewhere with usable data, rather
        # than disappearing from the categorized view entirely.
        found_ids = {item.get("id") for item in cat_list}
        for e in emails:
            if e["id"] not in found_ids:
                categories.setdefault("Primary", []).append({
                    **e,
                    "category_reason": "Uncategorised",
                    **_fallback_priority_for(e),
                })

        # Sort each category's emails by priority score, highest first, so
        # the merged Categorize view doubles as a priority-ranked inbox —
        # this is the actual replacement for the old standalone Priority page.
        for cat in categories:
            categories[cat].sort(key=lambda e: e.get("priority_score", 0), reverse=True)

        return jsonify({"categories": categories, "source": "ai_agent"})

    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#         AI SUMMARIZE — STREAMING SSE
# ==========================================

@app.route("/api/summarize/stream", methods=["POST"])
def api_summarize_stream():
    """
    Streaming summarize endpoint using Server-Sent Events (SSE).
    Sends chunks as they arrive from Groq, enabling typewriter effect on frontend.
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json() or {}
    email_text = data.get("emails_text", "")
    email_id   = data.get("email_id", None)   # if summarizing a single email

    if not email_text and not email_id:
        return jsonify({"error": "No content to summarize"}), 400
    if not GROQ_API_KEY and not NVIDIA_API_KEY:
        return jsonify({"error": "No AI provider configured — set GROQ_API_KEY and/or NVIDIA_API_KEY"}), 500

    if email_id:
        # Single email summary
        system_prompt = (
            "You are MailMind. Summarise this single email concisely in 3-5 bullet points. "
            "Include: what it's about, who sent it, what action (if any) is needed, and any deadline. "
            "Use emoji bullets for visual clarity. Keep it under 150 words."
        )
        user_content = email_text
    else:
        # Full inbox digest
        system_prompt = (
            "You are MailMind, an advanced email AI assistant. Write a structured daily digest.\n\n"
            "Format your response in clear sections:\n"
            "## 🔴 Time-Sensitive (needs action today)\n"
            "## 📋 Categories\n"
            "Group emails into: Job/Placements, Finance, Social, Promotions, Security, Updates, Newsletters\n"
            "For each category, list 1-2 line summaries.\n"
            "## ✅ Action Checklist\n"
            "End with exactly 3 critical action items the user must do.\n\n"
            "Be concise. Use emoji. Total response under 400 words."
        )
        user_content = email_text

    def generate():
        # Try primary provider first; if it fails with a 429/413 BEFORE any
        # bytes have been streamed to the client, silently retry the whole
        # request against the fallback provider. Once streaming has actually
        # started we can't retroactively swap providers mid-stream (the
        # client already has partial content), so a failure after that point
        # just surfaces as an ERROR event same as before.
        providers_to_try = [LLM_PROVIDERS["primary"]]
        if LLM_PROVIDERS["fallback"]["key"]:
            providers_to_try.append(LLM_PROVIDERS["fallback"])

        last_err = None
        for provider in providers_to_try:
            if not provider["key"]:
                continue
            try:
                payload = {
                    "model": provider["model"],
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_content},
                    ],
                    "stream": True,
                }
                headers = {"Authorization": f"Bearer {provider['key']}", "Content-Type": "application/json"}
                if provider["name"] == "openrouter":
                    headers["HTTP-Referer"] = "https://mailmind.local"
                    headers["X-Title"] = "MailMind"

                resp = requests.post(
                    provider["url"],
                    headers=headers,
                    json=payload,
                    timeout=60,
                    stream=True,
                )

                if resp.status_code in (429, 413):
                    # Nothing streamed yet — safe to fall through to the
                    # next provider in providers_to_try.
                    print(f"[summarize/stream] {provider['name']} returned {resp.status_code} — trying next provider")
                    last_err = f"{provider['name']} returned {resp.status_code}"
                    continue

                resp.raise_for_status()

                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data: "):
                            chunk_data = decoded[6:]
                            if chunk_data == "[DONE]":
                                yield "data: [DONE]\n\n"
                                return
                            try:
                                chunk_json = json.loads(chunk_data)
                                delta = chunk_json["choices"][0]["delta"].get("content", "")
                                if delta:
                                    # Escape newlines for SSE
                                    safe = delta.replace("\n", "\\n")
                                    yield f"data: {safe}\n\n"
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
                return  # streamed successfully from this provider — done
            except requests.exceptions.RequestException as e:
                last_err = str(e)
                continue

        yield f"data: ERROR:{last_err or 'All providers unavailable'}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":   "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ==========================================
#            SEARCH API
# ==========================================

@app.route("/api/search")
def api_search():
    """
    Server-side Gmail search. The frontend sends the query it already used
    for client-side Fuse.js filtering; this route hits the Gmail API so
    results aren't limited to whatever is already loaded on the page.

    Query params:
      q      - Gmail search query (same syntax as the Gmail search box)
      folder - optional view filter (unread, important, meetings)
      max    - max results to return (default 30, cap 50)
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    q = request.args.get("q", "").strip()
    folder = request.args.get("folder", "").strip()
    max_results = min(int(request.args.get("max", 30)), 50)

    if not q:
        return jsonify({"emails": [], "total": 0})

    service = get_gmail_service()
    try:
        gmail_q = q
        if folder == "unread":
            gmail_q = f"is:unread {q}"
        elif folder == "important":
            gmail_q = f"is:important {q}"
        elif folder == "meetings":
            gmail_q = f"(meeting OR interview OR call OR zoom OR meet) {q}"
        elif folder == "sent":
            gmail_q = f"in:sent {q}"

        results = service.users().messages().list(
            userId="me", q=gmail_q, maxResults=max_results,
        ).execute()
        messages = results.get("messages", [])
        if not messages:
            return jsonify({"emails": [], "total": 0})

        ids = [m["id"] for m in messages]
        details, _errors = fetch_messages_batch(service, ids, fmt="metadata", metadata_headers=EMAIL_METADATA_HEADERS)

        emails = []
        for msg_id in ids:
            detail = details.get(msg_id)
            if not detail:
                continue
            emails.append(build_email_obj(msg_id, detail))

        return jsonify({"emails": emails, "total": len(emails), "source": "gmail_search"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#            SENT MAIL API
# ==========================================

@app.route("/api/emails/sent")
def api_sent_emails():
    """
    Fetch emails from the user's real Gmail SENT label. Since this reads the
    same Gmail account the user authenticated with, anything sent through
    MailMind (or Gmail itself, or any other client) shows up here — it's not
    a separate local store, it's the actual Sent folder.

    - ?max=N  sets how many to fetch (default 30, cap 100)
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    service = get_gmail_service()
    max_results = min(int(request.args.get("max", 30)), 100)

    try:
        results = service.users().messages().list(
            userId="me", labelIds=["SENT"], maxResults=max_results,
        ).execute()
        messages = results.get("messages", [])
        if not messages:
            return jsonify({"emails": [], "total": 0})

        ids = [m["id"] for m in messages]
        details, _errors = fetch_messages_batch(service, ids, fmt="metadata", metadata_headers=EMAIL_METADATA_HEADERS)

        emails = []
        for msg_id in ids:
            detail = details.get(msg_id)
            if not detail:
                continue
            obj = build_email_obj(msg_id, detail)
            # For sent mail, "to" matters more than "from" in the UI, but we
            # keep build_email_obj's shape and just make sure "to" is present.
            emails.append(obj)

        return jsonify({"emails": emails, "total": len(emails)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#            COMPOSE / SEND NEW MAIL API
# ==========================================

@app.route("/api/compose", methods=["POST"])
def api_compose():
    """
    Send a brand-new email (not a reply to an existing thread).

    Expects JSON body:
      {
        "to":      "<recipient email address, comma-separated for multiple>",
        "cc":      "<optional cc addresses>",
        "subject": "<subject line>",
        "body":    "<plain text body>"
      }

    Requires the gmail.send scope -- returns 403 with needs_reauth=true if
    the current credentials don't have it, same as /api/reply, so the
    frontend can redirect the user to /login/with-send.

    Because this calls the real Gmail API's messages.send, the sent message
    lands in the account's actual SENT label — it will show up in real
    Gmail's Sent folder too, not just inside MailMind.
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if not has_send_scope():
        return jsonify({
            "error": "Send permission not granted. Please re-authenticate.",
            "needs_reauth": True,
            "reauth_url": url_for("login_with_send", _external=True),
        }), 403

    data = request.get_json() or {}
    to      = data.get("to", "").strip()
    cc      = data.get("cc", "").strip()
    subject = data.get("subject", "").strip()
    body    = data.get("body", "").strip()

    if not to or not subject or not body:
        return jsonify({"error": "Missing 'to', 'subject', or 'body'"}), 400

    import email.mime.text as _mime_text
    msg = _mime_text.MIMEText(body, "plain", "utf-8")
    msg["To"]      = to
    if cc:
        msg["Cc"] = cc
    msg["Subject"] = subject

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

    service = get_gmail_service()
    try:
        sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return jsonify({"success": True, "id": sent.get("id"), "threadId": sent.get("threadId")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#            REPLY 
# ==========================================

@app.route("/api/reply", methods=["POST"])
def api_reply():
    """
    Send a reply to an existing email thread.

    Expects JSON body:
      {
        "thread_id":    "<Gmail thread id>",
        "message_id":   "<RFC 2822 Message-ID header of the email being replied to>",
        "to":           "<recipient email address>",
        "subject":      "<reply subject (Re: original)>",
        "body":         "<plain text reply body>"
      }

    Requires the gmail.send scope -- returns 403 with needs_reauth=true if
    the current credentials don't have it so the frontend can redirect the
    user to /login/with-send.
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if not has_send_scope():
        return jsonify({
            "error": "Send permission not granted. Please re-authenticate.",
            "needs_reauth": True,
            "reauth_url": url_for("login_with_send", _external=True),
        }), 403

    data = request.get_json() or {}
    thread_id  = data.get("thread_id", "")
    message_id = data.get("message_id", "")
    to         = data.get("to", "").strip()
    subject    = data.get("subject", "")
    body       = data.get("body", "").strip()

    if not to or not body:
        return jsonify({"error": "Missing 'to' or 'body'"}), 400

    if not subject.lower().startswith("re:"):
        subject = "Re: " + subject

    import email.mime.text as _mime_text
    msg = _mime_text.MIMEText(body, "plain", "utf-8")
    msg["To"]      = to
    msg["Subject"] = subject
    if message_id:
        msg["In-Reply-To"] = message_id
        msg["References"]  = message_id

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

    service = get_gmail_service()
    try:
        send_body = {"raw": raw}
        if thread_id:
            send_body["threadId"] = thread_id

        sent = service.users().messages().send(userId="me", body=send_body).execute()
        return jsonify({"success": True, "id": sent.get("id"), "threadId": sent.get("threadId")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================
#         CREATE MEETING (Calendar + Invite)
# ==========================================

@app.route("/api/create-meeting", methods=["POST"])
def api_create_meeting():
    """
    Create a meeting: makes a real Google Calendar event (so it shows up on
    the user's actual Google Calendar), optionally with an auto-generated
    Google Meet link, and optionally emails an invite to the attendee(s)
    using the existing Gmail send pipeline.

    Expects JSON body:
      {
        "platform":   "google_meet" | "teams" | "zoom" | "skype" | "custom",
        "custom_link": "<url>"        -- required if platform is teams/zoom/skype/custom
        "title":      "<meeting title>",
        "description": "<optional notes>",
        "start":      "<ISO 8601 datetime, e.g. 2026-06-25T15:30:00>",
        "duration_minutes": 30,
        "attendees":  "<comma-separated email addresses>",
        "send_invite_email": true|false   -- whether to also send a Gmail invite
      }

    Requires the calendar.events scope -- returns 403 with needs_reauth=true
    if missing, same pattern as /api/reply and /api/compose, so the frontend
    can redirect to /login/with-calendar.

    If send_invite_email is true, also requires gmail.send (covered by the
    same calendar re-auth flow, since SCOPES_WITH_CALENDAR is a superset of
    SCOPES_WITH_SEND).
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if not has_calendar_scope():
        return jsonify({
            "error": "Calendar permission not granted. Please re-authenticate.",
            "needs_reauth": True,
            "reauth_url": url_for("login_with_calendar", _external=True),
        }), 403

    data = request.get_json() or {}
    platform          = data.get("platform", "google_meet").strip()
    custom_link       = data.get("custom_link", "").strip()
    title             = data.get("title", "").strip() or "Meeting"
    description       = data.get("description", "").strip()
    start_str         = data.get("start", "").strip()
    duration_minutes  = int(data.get("duration_minutes", 30) or 30)
    attendees_raw     = data.get("attendees", "").strip()
    send_invite_email = bool(data.get("send_invite_email", True))

    if not start_str:
        return jsonify({"error": "Missing 'start' date/time"}), 400
    if platform != "google_meet" and not custom_link:
        return jsonify({"error": f"Missing meeting link for platform '{platform}'"}), 400

    attendee_emails = [a.strip() for a in attendees_raw.split(",") if a.strip()]

    # Parse the start datetime. Frontend sends a naive local datetime string
    # (from an <input type="datetime-local">) — we treat it as being in the
    # user's own timezone and let Google Calendar attach the IANA timezone
    # name, rather than guessing a UTC offset ourselves.
    try:
        from datetime import datetime, timedelta
        start_dt = datetime.fromisoformat(start_str)
    except ValueError:
        return jsonify({"error": "Invalid 'start' datetime format"}), 400
    end_dt = start_dt + timedelta(minutes=duration_minutes)

    user_tz = data.get("timezone", "").strip() or "UTC"

    calendar_service = get_calendar_service()

    event_body = {
        "summary":     title,
        "description": description,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": user_tz},
        "end":   {"dateTime": end_dt.isoformat(),   "timeZone": user_tz},
    }
    if attendee_emails:
        event_body["attendees"] = [{"email": addr} for addr in attendee_emails]

    conference_request_id = None
    if platform == "google_meet":
        # Asking Calendar to generate a Google Meet link requires a
        # conferenceData request with a unique requestId per call.
        import uuid
        conference_request_id = uuid.uuid4().hex
        event_body["conferenceData"] = {
            "createRequest": {
                "requestId": conference_request_id,
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        }

    try:
        created_event = calendar_service.events().insert(
            calendarId="primary",
            body=event_body,
            conferenceDataVersion=1 if platform == "google_meet" else 0,
            sendUpdates="all" if attendee_emails else "none",
        ).execute()
    except HttpError as e:
        # This is the case from the screenshot: our session bookkeeping
        # (has_calendar_scope) thought the calendar.events scope was granted,
        # but Google's server is rejecting the actual access token for
        # lacking that permission. That mismatch happens when the scope was
        # never really granted server-side (Calendar API not enabled in the
        # Google Cloud project, or calendar.events not added to the OAuth
        # consent screen's scope list) even though the consent flow appeared
        # to succeed. Force a fresh, explicit re-consent rather than failing
        # with a confusing raw 500.
        reason = ""
        try:
            err_json = json.loads(e.content.decode("utf-8"))
            reason = err_json.get("error", {}).get("errors", [{}])[0].get("reason", "")
        except Exception:
            pass
        status = getattr(e.resp, "status", None)
        if status == 403 and ("insufficient" in reason.lower() or "insufficient" in str(e).lower()):
            return jsonify({
                "error": "Google rejected the calendar permission on this account. "
                         "This usually means the Calendar API isn't enabled in the Google "
                         "Cloud project, or the calendar.events scope isn't added to the "
                         "OAuth consent screen — re-authenticating won't fix it until that's "
                         "corrected in Cloud Console. Reconnect below once it is.",
                "needs_reauth": True,
                "reauth_url": url_for("login_with_calendar", _external=True),
            }), 403
        return jsonify({"error": f"Failed to create calendar event: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Failed to create calendar event: {e}"}), 500

    # Work out the actual meeting link to show/send: the freshly generated
    # Meet link for google_meet, or whatever link the user supplied for the
    # other platforms.
    if platform == "google_meet":
        meeting_link = created_event.get("hangoutLink", "")
        if not meeting_link:
            # Fall back to scanning conferenceData entry points if hangoutLink
            # wasn't populated for some reason.
            entry_points = created_event.get("conferenceData", {}).get("entryPoints", [])
            video_entry = next((e for e in entry_points if e.get("entryPointType") == "video"), None)
            meeting_link = video_entry.get("uri", "") if video_entry else ""
    else:
        meeting_link = custom_link

    platform_labels = {
        "google_meet": "Google Meet",
        "teams":       "Microsoft Teams",
        "zoom":        "Zoom",
        "skype":       "Skype",
        "custom":      "Video Call",
    }
    platform_label = platform_labels.get(platform, "Video Call")

    email_result = None
    if send_invite_email and attendee_emails:
        if not has_send_scope():
            # Shouldn't normally happen since SCOPES_WITH_CALENDAR includes
            # gmail.send, but guard anyway for accounts with stale tokens.
            email_result = {"sent": False, "error": "Missing gmail.send scope"}
        else:
            when_str = start_dt.strftime("%A, %d %B %Y at %I:%M %p").lstrip("0").replace(" 0", " ")
            invite_subject = f"Meeting Invite: {title}"
            invite_body = (
                f"You're invited to a meeting.\n\n"
                f"Title: {title}\n"
                f"When: {when_str} ({user_tz})\n"
                f"Platform: {platform_label}\n"
                f"Join link: {meeting_link}\n\n"
                + (f"Notes: {description}\n\n" if description else "")
                + "This invite was also added to the calendar.\n\n"
                f"— Sent via MailMind"
            )
            import email.mime.text as _mime_text
            msg = _mime_text.MIMEText(invite_body, "plain", "utf-8")
            msg["To"]      = ", ".join(attendee_emails)
            msg["Subject"] = invite_subject
            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")

            gmail_service = get_gmail_service()
            try:
                sent = gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()
                email_result = {"sent": True, "id": sent.get("id")}
            except Exception as e:
                email_result = {"sent": False, "error": str(e)}

    return jsonify({
        "success":       True,
        "event_id":      created_event.get("id"),
        "event_link":    created_event.get("htmlLink"),   # link to the event on Google Calendar
        "meeting_link":  meeting_link,
        "platform":      platform_label,
        "start":         start_dt.isoformat(),
        "end":           end_dt.isoformat(),
        "email_result":  email_result,
    })

# ==========================================
#   AGENTIC AI LAYER (LangChain + Groq)
#   Phases 2-7 of MailMind_Agentic_AI_Plan.docx
#
#   NOTE: requires the pre-1.0 langchain line —
#   see requirements.txt. langchain>=1.0 removed
#   create_tool_calling_agent / AgentExecutor.
# ==========================================

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq
# ChatOpenAI works with any OpenAI-compatible endpoint, including
# OpenRouter — used as the fallback LLM for the agentic chat when Groq's
# rate limit / token budget is exhausted (see build_agent_executor below).
from langchain_openai import ChatOpenAI


# ---- Phase 2: plain core functions ----

def core_get_important_emails(gmail_service, limit=10):
    """Returns a list of important-email dicts. Same logic as /api/agent/important."""
    results = gmail_service.users().messages().list(userId="me", maxResults=AGENT_EMAIL_FETCH_POOL).execute()
    messages = results.get("messages", [])
    if not messages:
        return []

    ids = [m["id"] for m in messages]
    details, _errors = fetch_messages_batch(gmail_service, ids, fmt="metadata", metadata_headers=EMAIL_METADATA_HEADERS)

    emails = []
    for msg_id in ids:
        detail = details.get(msg_id)
        if not detail:
            continue
        base = build_email_obj(msg_id, detail)
        base["gmail_imp"] = "IMPORTANT" in detail.get("labelIds", [])
        emails.append(base)

    if not emails:
        return []

    ai_emails = emails[:MAX_EMAILS_FOR_AI]
    email_list_text = "\n\n".join([
        f"[{i+1}] ID:{e['id']}\nFrom: {trim_field(e['from'], 60)}\nSubject: {trim_field(e['subject'], 80)}\nDate: {e['date']}\nSnippet: {trim_field(e['snippet'], 100)}"
        for i, e in enumerate(ai_emails)
    ])

    system_prompt = """You are an expert email importance classifier agent.
Classify as important if the email is from a real person needing a response,
contains a meeting/interview/offer, has an action required, is time-sensitive,
is from a known important sender, contains security alerts, or is Gmail-flagged
IMPORTANT. Do NOT classify newsletters, marketing, automated notices, or spam.

Respond ONLY with a valid JSON array. Each object:
{"id": "<email id>", "importance_score": <1-10>, "importance_reason": "<one sentence>", "action_needed": "<action or 'No action needed'>"}
Only include emails with importance_score >= 6. Sort by score descending."""

    important_list = groq_json_call(system_prompt, email_list_text)

    id_map = {e["id"]: e for e in emails}
    enriched = []
    for item in important_list:
        base = id_map.get(item.get("id"), {})
        if base:
            enriched.append({**base, **item})

    return enriched[:limit]


def core_search_emails(gmail_service, query, limit=5):
    """Search Gmail with standard Gmail search syntax and return matching
    emails with real message IDs, so the agent can resolve a vague
    description ('that mailmind discussion mail from Avikshit') into an
    actual id instead of asking the user for one."""
    results = gmail_service.users().messages().list(
        userId="me", q=query, maxResults=limit
    ).execute()
    messages = results.get("messages", [])
    if not messages:
        return []

    ids = [m["id"] for m in messages]
    details, _errors = fetch_messages_batch(
        gmail_service, ids, fmt="metadata", metadata_headers=EMAIL_METADATA_HEADERS
    )
    return [
        build_email_obj(mid, details[mid])
        for mid in ids if details.get(mid)
    ]


def core_summarize_email(email_text):
    """Returns a short 2-4 sentence summary of a single email's text. Same model call as /api/summarize."""
    system_prompt = (
        "You are MailMind. Summarise this single email concisely in 2-4 sentences. "
        "Include what it's about, who sent it, and what action (if any) is needed."
    )
    result = groq_call(system_prompt, email_text, max_tokens=300)
    return result["choices"][0]["message"]["content"]


def core_detect_meetings(gmail_service, limit=10):
    """Returns a list of meeting dicts with extracted date/time/link. Same logic as /api/agent/meetings."""
    query = "newer_than:30d (meeting OR interview OR call OR zoom OR meet OR webinar OR invite OR calendar OR schedule OR appointment OR standup OR sync OR demo OR presentation)"
    results = gmail_service.users().messages().list(userId="me", q=query, maxResults=20).execute()
    messages = results.get("messages", [])
    if not messages:
        return []

    ids = [m["id"] for m in messages]
    details, _errors = fetch_messages_batch(gmail_service, ids, fmt="full")

    emails = []
    for msg_id in ids:
        detail = details.get(msg_id)
        if not detail:
            continue
        emails.append(build_email_obj(msg_id, detail, include_body=True, body_char_limit=2000))

    if not emails:
        return []

    email_list_text = "\n\n".join([
        f"[{i+1}] ID:{e['id']}\nFrom: {trim_field(e['from'], 60)}\nSubject: {trim_field(e['subject'], 80)}\nEmail Date: {e['date']}\nBody preview: {trim_field(e.get('body', ''), 300)}"
        for i, e in enumerate(emails)
    ])

    system_prompt = """You are a calendar and meeting extraction AI agent.
For each email with a real meeting/call/interview/event, extract date, time,
type, link, organiser. Respond ONLY with a valid JSON array. Each object:
{"id":"<email id>","subject":"<subject>","meeting_type":"<interview|standup|zoom_call|webinar|client_call|team_meeting|other>",
"meeting_date":"<YYYY-MM-DD or null>","meeting_time":"<e.g. 3:00 PM IST, or null>",
"meeting_link":"<url or null>","organizer":"<who>","description":"<one sentence>"}
If no meetings found, return []."""

    meetings = groq_json_call(system_prompt, email_list_text)
    return meetings[:limit]


class CalendarPermissionError(Exception):
    """Raised when the user hasn't granted the calendar.events scope yet."""
    pass


class ContactsPermissionError(Exception):
    """Raised when the user hasn't granted the contacts.readonly scope yet."""
    pass


def _resolve_contacts_via_people_api(people_service, name_query, limit=5):
    """Real lookup against the user's actual Google Contacts. Returns a
    list of {"name": ..., "email": ...} for EVERY plausible match (up to
    limit), not just the first — so the caller can tell a single clean
    match apart from a genuine name collision (e.g. two contacts named
    'Anant') and decide whether to resolve automatically or ask the user."""
    try:
        results = people_service.people().searchContacts(
            query=name_query,
            readMask="names,emailAddresses",
        ).execute()
    except HttpError:
        return []

    matches = []
    for entry in results.get("results", [])[:limit]:
        person = entry.get("person", {})
        names = person.get("names", [])
        emails = person.get("emailAddresses", [])
        if emails:
            matches.append({
                "name": names[0].get("displayName", name_query) if names else name_query,
                "email": emails[0].get("value"),
            })
    return matches


def _resolve_contact_via_gmail_fallback(gmail_service, name_query, limit=5):
    """Fallback used only when the user hasn't granted contacts.readonly:
    scan recent Gmail headers for a From/To field containing the name.
    Less reliable than the People API (nicknames, CC-only mentions, name
    collisions all confuse it) but keeps the feature usable pre-reauth.
    Returns a single email address string, or None — this path doesn't
    attempt ambiguity detection since Gmail header scanning is already a
    weaker signal than a real Contacts match."""
    try:
        results = gmail_service.users().messages().list(
            userId="me", q=f'"{name_query}"', maxResults=limit
        ).execute()
        ids = [m["id"] for m in results.get("messages", [])]
        if not ids:
            return None
        details, _errors = fetch_messages_batch(gmail_service, ids, fmt="metadata", metadata_headers=["From", "To"])
        for msg_id in ids:
            detail = details.get(msg_id)
            if not detail:
                continue
            headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
            for field in ("From", "To"):
                val = headers.get(field, "")
                if name_query.lower() in val.lower():
                    m = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', val)
                    if m:
                        return m.group(0)
    except Exception:
        pass
    return None


def resolve_contact(people_service, gmail_service, has_contacts, name_query):
    """
    Resolve a name or group label to contact match(es).
    Prefers the real Google Contacts lookup (People API), which can return
    multiple matches for a name collision. Falls back to a heuristic Gmail
    header scan (single result, no ambiguity detection) only if the user
    hasn't granted contacts.readonly yet, or Contacts had no match.

    Returns one of:
      {"status": "single", "email": "...", "name": "..."}       — exactly one match, safe to use directly
      {"status": "ambiguous", "candidates": [{"name","email"}, ...]}  — 2+ matches, caller must ask the user
      {"status": "none"}                                         — no match anywhere
    """
    if has_contacts and people_service is not None:
        matches = _resolve_contacts_via_people_api(people_service, name_query)
        if len(matches) == 1:
            return {"status": "single", "email": matches[0]["email"], "name": matches[0]["name"]}
        if len(matches) > 1:
            return {"status": "ambiguous", "candidates": matches}
        # No Contacts match — fall through to Gmail header scan below.

    addr = _resolve_contact_via_gmail_fallback(gmail_service, name_query)
    if addr:
        return {"status": "single", "email": addr, "name": name_query}
    return {"status": "none"}


def core_create_calendar_event(calendar_service, gmail_service, has_calendar, has_send,
                                title, start_iso, duration_minutes=30,
                                attendees="", description="", timezone="UTC",
                                send_invite_email=True, platform="google_meet"):
    """
    Creates a real Google Calendar event (+ Meet link) and optionally emails
    an invite. Same logic as /api/create-meeting. Raises CalendarPermissionError
    if the calendar.events scope isn't granted, so the caller can trigger reauth.
    """
    if not has_calendar:
        raise CalendarPermissionError("Calendar permission not granted — user must re-authenticate at /login/with-calendar")

    from datetime import datetime, timedelta
    import uuid
    import base64 as _b64
    import email.mime.text as _mime_text

    start_dt = datetime.fromisoformat(start_iso)
    end_dt = start_dt + timedelta(minutes=duration_minutes)
    attendee_emails = [a.strip() for a in attendees.split(",") if a.strip()]

    event_body = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start_dt.isoformat(), "timeZone": timezone},
        "end":   {"dateTime": end_dt.isoformat(),   "timeZone": timezone},
    }
    if attendee_emails:
        event_body["attendees"] = [{"email": a} for a in attendee_emails]
    if platform == "google_meet":
        event_body["conferenceData"] = {
            "createRequest": {
                "requestId": uuid.uuid4().hex,
                "conferenceSolutionKey": {"type": "hangoutsMeet"},
            }
        }

    created_event = calendar_service.events().insert(
        calendarId="primary",
        body=event_body,
        conferenceDataVersion=1 if platform == "google_meet" else 0,
        sendUpdates="all" if attendee_emails else "none",
    ).execute()

    meeting_link = created_event.get("hangoutLink", "")

    email_sent = False
    if send_invite_email and attendee_emails and has_send:
        when_str = start_dt.strftime("%A, %d %B %Y at %I:%M %p").lstrip("0").replace(" 0", " ")
        body_text = (f"You're invited to a meeting.\n\nTitle: {title}\nWhen: {when_str} ({timezone})\n"
                     f"Join link: {meeting_link}\n\n" + (f"Notes: {description}\n\n" if description else "")
                     + "— Sent via MailMind")
        msg = _mime_text.MIMEText(body_text, "plain", "utf-8")
        msg["To"] = ", ".join(attendee_emails)
        msg["Subject"] = f"Meeting Invite: {title}"
        raw = _b64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
        gmail_service.users().messages().send(userId="me", body={"raw": raw}).execute()
        email_sent = True

    return {
        "event_id": created_event.get("id"),
        "event_link": created_event.get("htmlLink"),
        "meeting_link": meeting_link,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "invite_emailed": email_sent,
    }

def core_list_upcoming_events(calendar_service, max_results=10):
    events_result = calendar_service.events().list(
        calendarId="primary",
        timeMin=datetime.utcnow().isoformat() + "Z",
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime"
    ).execute()
    events = events_result.get("items", [])
    return [
        {
            "event_id": e["id"],
            "title": e.get("summary", "(no title)"),
            "start": e.get("start", {}).get("dateTime", e.get("start", {}).get("date")),
        }
        for e in events
    ]

def core_update_calendar_event(calendar_service, event_id, new_start_iso=None,
                                 duration_minutes=None, timezone="UTC"):
    event = calendar_service.events().get(calendarId="primary", eventId=event_id).execute()
    if new_start_iso:
        start_dt = datetime.fromisoformat(new_start_iso)
        end_dt = start_dt + timedelta(minutes=duration_minutes or 30)
        event["start"] = {"dateTime": start_dt.isoformat(), "timeZone": timezone}
        event["end"] = {"dateTime": end_dt.isoformat(), "timeZone": timezone}
    updated = calendar_service.events().patch(
        calendarId="primary", eventId=event_id, body=event, sendUpdates="all"
    ).execute()
    return {
        "event_id": updated.get("id"),
        "event_link": updated.get("htmlLink"),
        "meet_link": updated.get("hangoutLink"),
    }

def core_archive_emails(gmail_service, query, limit=50):
    """Archive (remove INBOX label from) all emails matching a Gmail search
    query. Returns how many were archived so the caller can show a count
    before/after confirmation."""
    results = gmail_service.users().messages().list(userId="me", q=query, maxResults=limit).execute()
    ids = [m["id"] for m in results.get("messages", [])]
    if not ids:
        return {"archived": 0, "query": query}
    gmail_service.users().messages().batchModify(
        userId="me", body={"ids": ids, "removeLabelIds": ["INBOX"]}
    ).execute()
    return {"archived": len(ids), "query": query}


def core_count_matching(gmail_service, query, cap=50):
    """Count emails matching a query without archiving them — used to show
    the user what a proposed archive action would affect before they confirm."""
    results = gmail_service.users().messages().list(userId="me", q=query, maxResults=cap).execute()
    return len(results.get("messages", []))


def core_draft_reply(gmail_service, email_id, instruction):
    """
    Fetch a specific email by id and draft a reply body via Groq, based on
    the user's instruction (e.g. 'thank him for the update'). Does NOT send
    anything — returns the drafted to/subject/body for the caller to hold
    as a pending action until the user confirms.
    """
    detail = gmail_service.users().messages().get(userId="me", id=email_id, format="metadata",
                                                    metadataHeaders=["From", "Subject"]).execute()
    headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
    from_header = headers.get("From", "")
    subject = headers.get("Subject", "(no subject)")
    snippet = detail.get("snippet", "")

    gen_prompt = (
        f"Write a short, professional email reply. Original email — "
        f"From: {from_header}, Subject: {subject}, Snippet: {snippet}. "
        f"Reply instruction from the user: {instruction}. "
        f"Output ONLY the reply body text, no subject line, no signature."
    )
    body_text = groq_call(
        "You write concise professional email replies.", gen_prompt, max_tokens=350
    )["choices"][0]["message"]["content"].strip()

    to_match = re.search(r'<([^>]+)>', from_header)
    to = to_match.group(1) if to_match else from_header
    reply_subject = subject if subject.lower().startswith("re:") else "Re: " + subject

    return {"email_id": email_id, "to": to, "subject": reply_subject, "body": body_text}


def core_compose_new_email(to, subject, instruction):
    """
    Draft a brand-new email (NOT a reply to an existing message) to a given
    recipient, based on a natural-language instruction like 'ask him for
    the updated resume' or 'invite her to the 5pm sync'. Does NOT send
    anything — returns to/subject/body for the caller to hold as a pending
    action until the user confirms, exactly like core_draft_reply does for
    replies.
    """
    gen_prompt = (
        f"Write a short, professional email. Recipient: {to}. "
        f"Subject line hint: {subject or '(pick a suitable subject yourself)'}. "
        f"What the email should say: {instruction}. "
        f"Output ONLY the email body text, no subject line, no signature."
    )
    body_text = groq_call(
        "You write concise professional emails.", gen_prompt, max_tokens=350
    )["choices"][0]["message"]["content"].strip()

    final_subject = subject.strip() if subject and subject.strip() else "Quick note"
    return {"to": to, "subject": final_subject, "body": body_text}


def core_send_drafted_reply(gmail_service, to, subject, body, thread_id=None):
    """Actually sends a reply previously drafted by core_draft_reply, once confirmed."""
    import email.mime.text as _mime_text
    msg = _mime_text.MIMEText(body, "plain", "utf-8")
    msg["To"], msg["Subject"] = to, subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    send_body = {"raw": raw}
    if thread_id:
        send_body["threadId"] = thread_id
    sent = gmail_service.users().messages().send(userId="me", body=send_body).execute()
    return {"sent": True, "to": to, "id": sent.get("id")}


# ==========================================
#   PENDING ACTIONS — structured confirmation
#   for destructive/send agent actions.
#
#   Tools that archive, reply, or send never
#   execute directly. They write a pending
#   action here and return a JSON description;
#   /api/agent/confirm/<id> executes it once
#   the user explicitly approves.
#
#   NOTE: in-memory + single-process. Fine for
#   one dev/small deployment; swap for a DB
#   table or Redis if you run >1 worker.
# ==========================================
import uuid as _uuid_mod

pending_actions = {}  # action_id -> {"type", "user_email", "created_at", ...action fields}
PENDING_ACTION_TTL_SECONDS = 15 * 60


def _prune_expired_actions():
    now = datetime.utcnow()
    expired = [
        aid for aid, a in pending_actions.items()
        if (now - a["created_at"]).total_seconds() > PENDING_ACTION_TTL_SECONDS
    ]
    for aid in expired:
        pending_actions.pop(aid, None)


def create_pending_action(action_type, user_email, **fields):
    _prune_expired_actions()
    action_id = _uuid_mod.uuid4().hex
    pending_actions[action_id] = {
        "type": action_type,
        "user_email": user_email,
        "created_at": datetime.utcnow(),
        **fields,
    }
    return action_id


# ---- Phase 3 + 4: wrap as LangChain tools (per-request factory, since
# tools need the current request's session-bound Gmail/Calendar services) ----

def build_agent_tools(gmail_service, calendar_service, people_service,
                       has_calendar, has_send, has_contacts, user_email):
    """
    Every tool returns a JSON string with a "status" field:
      "ok"                  -> normal successful result
      "needs_reauth"        -> missing scope; includes "reauth_url"
      "pending_confirmation"-> a destructive/send action was drafted, not
                                executed; includes "action_id" the user must
                                confirm via /api/agent/confirm/<action_id>
      "error"               -> tool-level failure; includes "message"

    This is deterministic and parsed by the route itself (not just left for
    the LLM to relay in prose), so the frontend can reliably show a reauth
    link or a Confirm button regardless of how the model phrases its
    final answer.
    """

    @tool
    def get_important_emails(limit: int = 10) -> str:
        """Fetch the user's important unread/flagged emails, AI-scored 1-10. Returns a JSON string list."""
        try:
            emails = core_get_important_emails(gmail_service, limit=limit)
            return json.dumps({"status": "ok", "emails": emails})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error fetching important emails: {e}"})

    @tool
    def summarize_email(email_text: str) -> str:
        """Summarize a single email's text (subject/from/body) in 2-4 sentences."""
        try:
            summary = core_summarize_email(email_text)
            return json.dumps({"status": "ok", "summary": summary})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error summarizing: {e}"})

    @tool
    def search_emails(query: str, limit: int = 5) -> str:
        """
        Search the user's Gmail to find a specific email's real message ID.
        Use Gmail search syntax — e.g. 'from:avikshit mailmind', 'subject:meeting invite'.
        ALWAYS call this to resolve a vague reference ('that mailmind discussion
        mail', 'the invite from Avikshit') to a real id BEFORE calling
        propose_reply or summarize_email — never ask the user for an email id
        directly. If you already have the id from an earlier tool result in
        this same conversation, reuse it instead of searching again.
        """
        try:
            emails = core_search_emails(gmail_service, query, limit=limit)
            return json.dumps({"status": "ok", "emails": emails})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error searching emails: {e}"})

    @tool
    def detect_meetings(limit: int = 10) -> str:
        """Scan recent emails for meetings/interviews/calls and extract date, time, and link. Returns a JSON string list."""
        try:
            meetings = core_detect_meetings(gmail_service, limit=limit)
            return json.dumps({"status": "ok", "meetings": meetings})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error detecting meetings: {e}"})

    @tool
    def find_contact_email(name_or_group: str) -> str:
        """
        Resolve a person's name or a group label (e.g. 'Rahul', 'HR', 'the
        team') to a real email address, using the user's Google Contacts
        (falls back to scanning Gmail history if Contacts access isn't
        granted). Use this before create_calendar_event or propose_reply
        whenever the user names someone instead of giving an exact address.

        Returns status 'ok' with a single email if there's exactly one
        match. Returns status 'ambiguous' with a list of candidates if
        multiple contacts share that name (e.g. two people named 'Anant')
        — in that case you MUST list every candidate's name and email to
        the user and ask them which one they mean; never guess.
        """
        result = resolve_contact(people_service, gmail_service, has_contacts, name_or_group)

        if result["status"] == "single":
            return json.dumps({"status": "ok", "name": name_or_group, "email": result["email"]})

        if result["status"] == "ambiguous":
            return json.dumps({
                "status": "ambiguous",
                "name_query": name_or_group,
                "candidates": result["candidates"],
                "message": f"Found {len(result['candidates'])} contacts matching "
                           f"'{name_or_group}'. List them for the user and ask which one they mean.",
            })

        return json.dumps({
            "status": "error",
            "message": f"Could not find an email address for '{name_or_group}'. "
                       f"Ask the user for it directly, or add it under Settings > Contact groups."
        })

    @tool
    def create_calendar_event(title: str, start_iso: str, duration_minutes: int = 30,
                               attendees: str = "", description: str = "",
                               timezone: str = "UTC") -> str:
        """
        Create a real Google Calendar event with an auto-generated Google Meet link,
        and email the invite to attendees. start_iso must be ISO 8601, e.g.
        '2026-07-10T15:30:00'. attendees is a comma-separated list of emails —
        resolve names to addresses with find_contact_email first if needed.
        """
        try:
            result = core_create_calendar_event(
                calendar_service, gmail_service, has_calendar, has_send,
                title=title, start_iso=start_iso, duration_minutes=duration_minutes,
                attendees=attendees, description=description, timezone=timezone,
            )
            return json.dumps({"status": "ok", **result})
        except CalendarPermissionError:
            return json.dumps({
                "status": "needs_reauth",
                "reauth_url": "/login/with-calendar",
                "message": "Calendar permission not granted yet.",
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error creating event: {e}"})
    @tool
    def list_upcoming_events(max_results: int = 10) -> str:
        """
        List the user's upcoming calendar events with their event_id, title, and start time.
        ALWAYS call this first to find the correct event_id before rescheduling,
        updating, or cancelling any meeting — never guess an event_id.
        """
        try:
            events = core_list_upcoming_events(calendar_service, max_results=max_results)
            return json.dumps({"status": "ok", "events": events})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error listing events: {e}"})

    @tool
    def update_calendar_event(event_id: str, new_start_iso: str = None,
                               duration_minutes: int = None, timezone: str = "UTC") -> str:
        """
        Reschedule/update an EXISTING calendar event by its event_id (never
        creates a new one). Get the event_id from list_upcoming_events first.
        Use this whenever the user says 'reschedule', 'move', 'update the time
        of', or 'change' an existing meeting.
        """
        try:
            result = core_update_calendar_event(
                calendar_service, event_id, new_start_iso=new_start_iso,
                duration_minutes=duration_minutes, timezone=timezone,
            )
            return json.dumps({"status": "ok", **result})
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error updating event: {e}"})
        
    @tool
    def propose_archive_emails(query: str) -> str:
        """
        Propose archiving (removing from inbox) all emails matching a Gmail
        search query, e.g. 'from:amazon.com' or 'category:promotions older_than:30d'.
        Does NOT archive anything yet — shows a count and returns an
        action_id the user must confirm before it actually runs.
        """
        try:
            count = core_count_matching(gmail_service, query)
            if count == 0:
                return json.dumps({"status": "ok", "message": f"No emails match '{query}' — nothing to archive."})
            action_id = create_pending_action("archive", user_email, query=query, count=count)
            return json.dumps({
                "status": "pending_confirmation",
                "action_id": action_id,
                "action_type": "archive",
                "query": query,
                "count": count,
                "message": f"Found {count} email(s) matching '{query}'. Ask the user to confirm before archiving.",
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error checking matching emails: {e}"})

    @tool
    def propose_reply(email_id: str, instruction: str) -> str:
        """
        Draft a reply to a specific email based on an instruction like
        'thank him for the update' or 'decline politely'.
        email_id MUST be a real Gmail message id — a hex string like
        '18f2a9c4b3d7e001', copied exactly from the "id" field of a result
        returned by search_emails, get_important_emails, or detect_meetings.
        NEVER pass a tool name, a person's name, an email address, or a
        description as email_id — if you don't already have a real id from
        an earlier tool result in this conversation, call search_emails
        first to get one.
        Does NOT send anything — returns a drafted to/subject/body and an
        action_id the user must confirm before it actually sends.
        """
        try:
            if not has_send:
                return json.dumps({
                    "status": "needs_reauth",
                    "reauth_url": "/login/with-send",
                    "message": "Send permission not granted yet.",
                })
            # Defensive guard: Gmail message ids are lowercase hex strings.
            # Catching an obviously-wrong id here (a tool name, a person's
            # name, an email address, etc.) avoids a confusing raw 400 from
            # the Gmail API and gives the model a clear, actionable message
            # instead — this is the fix for a real failure mode where the
            # model passed the literal string "search_emails" as email_id.
            if not re.fullmatch(r"[0-9a-fA-F]{10,}", (email_id or "").strip()):
                return json.dumps({
                    "status": "error",
                    "message": f"'{email_id}' is not a valid Gmail message id. "
                               f"Call search_emails first to find the real id "
                               f"(a hex string like '18f2a9c4b3d7e001'), then "
                               f"call propose_reply again with that exact id.",
                })
            draft = core_draft_reply(gmail_service, email_id, instruction)
            action_id = create_pending_action("reply", user_email, **draft)
            return json.dumps({
                "status": "pending_confirmation",
                "action_id": action_id,
                "action_type": "reply",
                "to": draft["to"],
                "subject": draft["subject"],
                "body_preview": draft["body"],
                "message": "Reply drafted. Ask the user to confirm before sending.",
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error drafting reply: {e}"})

    @tool
    def propose_new_email(to_email: str, instruction: str, subject: str = "") -> str:
        """
        Draft a brand-new email to send to someone — use this when the user
        wants to SEND/WRITE a fresh email (not reply to an existing one),
        e.g. 'send a mail to Avikshit asking about the deadline'.
        to_email MUST be a real, exact email address — if the user only gave
        a name, call find_contact_email FIRST to resolve it. If
        find_contact_email returns 'ambiguous' (2+ people share that name),
        you MUST list every candidate's name and email address and ask the
        user to pick the exact one — never guess between
        'a.avikshit@gmail.com' and 'avikshit@gmail.com', for example — and
        do not call this tool until the user has chosen.
        Does NOT send anything — it only drafts the email and returns an
        action_id the user must confirm before it actually sends.
        """
        try:
            if not has_send:
                return json.dumps({
                    "status": "needs_reauth",
                    "reauth_url": "/login/with-send",
                    "message": "Send permission not granted yet.",
                })
            if "@" not in (to_email or ""):
                return json.dumps({
                    "status": "error",
                    "message": f"'{to_email}' doesn't look like a real email address. "
                               f"Call find_contact_email first to resolve the name to an "
                               f"exact address, then call propose_new_email again with that address.",
                })
            draft = core_compose_new_email(to_email, subject, instruction)
            action_id = create_pending_action("compose", user_email, **draft)
            return json.dumps({
                "status": "pending_confirmation",
                "action_id": action_id,
                "action_type": "compose",
                "to": draft["to"],
                "subject": draft["subject"],
                "body_preview": draft["body"],
                "message": "Email drafted. Ask the user to confirm before sending.",
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": f"Error drafting email: {e}"})

    return [
        get_important_emails, summarize_email, detect_meetings,
        find_contact_email, create_calendar_event,
        propose_archive_emails, propose_reply, propose_new_email, search_emails,
    ]


# ---- Phase 5: build the AgentExecutor ----

def build_agent_executor(tools, provider_key="primary"):
    """
    provider_key selects which entry in LLM_PROVIDERS to build the LLM
    from — "primary" (Groq) by default, "fallback" (OpenRouter → GLM-5.2)
    when the caller has detected the primary is rate-limited/exhausted.
    Both ChatGroq and ChatOpenAI implement the same LangChain chat-model
    interface, so create_tool_calling_agent/AgentExecutor work identically
    regardless of which one is passed in.
    """
    provider = LLM_PROVIDERS[provider_key]
    # temperature low: we want reliable tool-call decisions, not creative
    # wording. max_tokens capped well under the model's context window —
    # plenty for a tool-calling loop's input+scratchpad+output.
    if provider["name"] == "groq":
        llm = ChatGroq(model=provider["model"], api_key=provider["key"], temperature=0.1, max_tokens=2048)
    else:
        # NVIDIA NIM (or any other OpenAI-compatible provider) via ChatOpenAI
        # with a custom base_url — same request/response shape as OpenAI's
        # own API, which is what NVIDIA's endpoint mirrors.
        llm = ChatOpenAI(
            model=provider["model"],
            api_key=provider["key"],
            base_url="https://integrate.api.nvidia.com/v1",
            temperature=0.1,
            max_tokens=2048,
        )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are MailMind's email assistant. You can fetch important emails, "
         "summarize a specific email, detect meetings in the inbox, search "
         "Gmail to find a specific email's real message id, resolve a name "
         "to a real email address, create real calendar events with Meet "
         "links + invite emails, propose archiving emails matching a search "
         "query, propose a reply to a specific email, and propose a brand-new "
         "email to someone (not a reply). "
         "When the user asks to SEND/WRITE a fresh email to a person ('send a "
         "mail to Avikshit', 'email Priya about...') — this is NOT a reply, "
         "use propose_new_email, never propose_reply. Always call "
         "find_contact_email first to turn the name into a real address "
         "before calling propose_new_email. "
         "Only use the tools provided — if a request has no matching tool "
         "(e.g. unrelated to email/calendar), say so plainly instead of guessing. "
         "Never ask the user for an email id. When they refer to an email by "
         "description ('that mailmind discussion mail', 'the invite from "
         "Avikshit'), call search_emails with keywords from their description "
         "(combine with find_contact_email first if they named a person) to "
         "find it yourself, then use the id from the top matching result. "
         "Only ask the user to disambiguate if search_emails returns several "
         "plausible matches and you genuinely can't tell which one they mean. "
         "When the user names a person or group instead of giving an exact "
         "email address, call find_contact_email first. If find_contact_email "
         "returns status 'ambiguous' (multiple contacts share that name), list "
         "every candidate by name AND their exact email address (e.g. "
         "'a.avikshit@gmail.com' vs 'avikshit@gmail.com') in your reply and "
         "ask the user which exact address they mean — never guess between "
         "two people with the same name, and never proceed to "
         "propose_new_email/propose_reply/create_calendar_event until the "
         "user has picked one by address. "
         "propose_archive_emails, propose_reply, and propose_new_email never "
         "execute immediately — they draft the action and wait for user "
         "confirmation, so you do not need to ask for confirmation yourself "
         "before calling them; just call them and report back what was "
         "found/drafted, and tell the user to confirm to actually send/archive it. "
         "For creating calendar events, if the user hasn't given an exact "
         "date/time, ask them to confirm before calling the tool, since that "
         "one does execute directly. "
         "When you list specific emails, always mention the sender's name and "
         "email address and the subject in your final answer — later turns in "
         "this conversation may need to refer back to 'him'/'her'/'that email' "
         "and only your own prior answers (not raw tool output) carry forward. "
         "If a tool call returns status 'error', relay that error honestly to "
         "the user and suggest a next step (e.g. try a different search "
         "query) — never claim an action succeeded when a tool reported an "
         "error. "
         "Give a clear, concise natural-language final answer, not raw JSON."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    # return_intermediate_steps lets the route inspect what each tool
    # actually returned (status: needs_reauth / pending_confirmation) rather
    # than relying on the LLM's final prose to carry that signal correctly.
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=6,
                          return_intermediate_steps=True)


def _extract_tool_signal(intermediate_steps):
    """
    Scan the agent's tool-call observations for the first structured
    needs_reauth / pending_confirmation / ambiguous / error status, so the
    API response can carry it deterministically (frontend renders a reauth
    link, Confirm button, or contact picker) instead of depending on the
    model echoing it correctly in text.
    """
    for _action, observation in intermediate_steps:
        if not isinstance(observation, str):
            continue
        try:
            parsed = json.loads(observation)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed.get("status") in (
            "needs_reauth", "pending_confirmation", "error", "ambiguous"
        ):
            return parsed
    return None


CHAT_HISTORY_MAX_TURNS = 10  # keep last N user+assistant pairs (20 messages)


# ---- Phase 6 + 7: wire the Flask route ----

@app.route("/api/agent/chat", methods=["POST"])
def api_agent_chat():
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401
    if not GROQ_API_KEY and not NVIDIA_API_KEY:
        return jsonify({"error": "No AI provider configured — set GROQ_API_KEY and/or NVIDIA_API_KEY"}), 500

    user_prompt = (request.get_json() or {}).get("prompt", "").strip()
    if not user_prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    gmail_service = get_gmail_service()
    calendar_service = get_calendar_service()
    people_service = get_people_service() if has_contacts_scope() else None

    tools = build_agent_tools(
        gmail_service, calendar_service, people_service,
        has_calendar=has_calendar_scope(),
        has_send=has_send_scope(),
        has_contacts=has_contacts_scope(),
        user_email=session.get("user_email"),
    )

    # Rebuild LangChain message history from what we stored in the session
    # on the previous turn(s), so pronouns/context carry across requests —
    # this is the real fix for "reply thanking him" style follow-ups.
    stored_history = session.get("agent_chat_history", [])
    lc_history = [
        (HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]))
        for m in stored_history
    ]

    def _looks_like_rate_limit(exc):
        """Detect a rate-limit/quota-shaped error from the LangChain LLM
        client's raised exception, since ChatGroq/ChatOpenAI wrap the
        underlying HTTP error rather than raising our own GroqRateLimitError."""
        s = str(exc).lower()
        return any(tok in s for tok in ("rate_limit", "429", "quota", "too many requests"))

    # Try the primary provider (Groq) first; on a rate-limit-shaped error,
    # silently rebuild the executor against the fallback provider
    # (OpenRouter → GLM-5.2) and retry once — no extra action needed from
    # the user, the switch just happens.
    provider_used = "primary"
    try:
        agent_executor = build_agent_executor(tools, provider_key="primary")
        result = agent_executor.invoke({"input": user_prompt, "chat_history": lc_history})
    except Exception as primary_exc:
        if _looks_like_rate_limit(primary_exc) and NVIDIA_API_KEY:
            print(f"[api_agent_chat] primary provider rate-limited ({primary_exc}) — falling back to OpenRouter/GLM-5.2")
            provider_used = "fallback"
            try:
                agent_executor = build_agent_executor(tools, provider_key="fallback")
                result = agent_executor.invoke({"input": user_prompt, "chat_history": lc_history})
            except Exception as fallback_exc:
                return jsonify({"error": f"Both AI providers failed: {fallback_exc}"}), 500
        else:
            return jsonify({"error": f"Agent error: {primary_exc}"}), 500

    try:
        output_text = result["output"]

        stored_history.append({"role": "user", "content": user_prompt})
        stored_history.append({"role": "assistant", "content": output_text})
        session["agent_chat_history"] = stored_history[-(CHAT_HISTORY_MAX_TURNS * 2):]

        response_payload = {"response": output_text}
        if provider_used == "fallback":
            # Purely informational — lets the frontend show a subtle
            # "using backup AI" indicator if it wants to, without being
            # required to.
            response_payload["provider"] = "fallback"

        signal = _extract_tool_signal(result.get("intermediate_steps", []))
        if signal:
            if signal["status"] == "needs_reauth":
                response_payload["needs_reauth"] = True
                response_payload["reauth_url"] = signal.get("reauth_url")
            elif signal["status"] == "pending_confirmation":
                response_payload["pending_action"] = {
                    k: v for k, v in signal.items() if k != "status"
                }
            elif signal["status"] == "error":
                # Surfaced separately from the model's own prose so the
                # frontend can show a clear error state even if the model
                # tried to spin a failed tool call into a false success.
                response_payload["tool_error"] = signal.get("message")
            elif signal["status"] == "ambiguous":
                # Multiple contacts share the requested name — the model's
                # prose already lists them, but this structured field lets
                # the frontend render real clickable choice buttons instead
                # of relying on the user to retype a name correctly.
                response_payload["contact_candidates"] = {
                    "name_query": signal.get("name_query"),
                    "candidates": signal.get("candidates", []),
                }

        return jsonify(response_payload)
    except Exception as e:
        return jsonify({"error": f"Agent error: {e}"}), 500


@app.route("/api/agent/confirm/<action_id>", methods=["POST"])
def api_agent_confirm(action_id):
    """
    Executes a pending action (archive or reply) previously drafted by the
    agent. The frontend calls this when the user taps Confirm on the
    pending_action card the chat UI rendered from /api/agent/chat's response.
    """
    if "credentials" not in session:
        return jsonify({"error": "Not authenticated"}), 401

    action = pending_actions.get(action_id)
    if not action:
        return jsonify({"error": "This action has expired or was already handled. Please ask the agent again."}), 404
    if action.get("user_email") != session.get("user_email"):
        return jsonify({"error": "Not authorized for this action"}), 403

    gmail_service = get_gmail_service()

    try:
        if action["type"] == "archive":
            result = core_archive_emails(gmail_service, action["query"])
        elif action["type"] == "reply":
            result = core_send_drafted_reply(gmail_service, action["to"], action["subject"], action["body"])
        elif action["type"] == "compose":
            result = core_send_drafted_reply(gmail_service, action["to"], action["subject"], action["body"])
        else:
            return jsonify({"error": f"Unknown action type '{action['type']}'"}), 400

        pending_actions.pop(action_id, None)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent/confirm/<action_id>", methods=["DELETE"])
def api_agent_cancel(action_id):
    """Discards a pending action without executing it (user tapped Cancel)."""
    action = pending_actions.get(action_id)
    if action and action.get("user_email") == session.get("user_email"):
        pending_actions.pop(action_id, None)
    return jsonify({"success": True})


@app.route("/api/agent/reset", methods=["POST"])
def api_agent_reset():
    """Clears the stored chat history — used by a 'New conversation' button."""
    session.pop("agent_chat_history", None)
    return jsonify({"success": True})

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

# Fixed time for everyone, in IST (Asia/Kolkata) — NOT server-local time.
# Previously CronTrigger had no timezone, so it silently ran on whatever
# timezone the host machine/container was set to (often UTC), which meant
# "7 PM" in the code could actually fire at a completely different clock
# time in India. Pinning the timezone here makes DIGEST_HOUR/DIGEST_MINUTE
# mean IST wall-clock time regardless of where the server actually runs.
DIGEST_TIMEZONE = ZoneInfo("Asia/Kolkata")
DIGEST_HOUR   = int(os.getenv("DIGEST_HOUR", 16))    # 0 = midnight, IST
DIGEST_MINUTE = int(os.getenv("DIGEST_MINUTE", 2))

DIGEST_SYSTEM_PROMPT = (
    "You are MailMind, an advanced email AI assistant. Write a structured daily digest.\n\n"
    "Format your response in clear sections:\n"
    "## 🔴 Time-Sensitive (needs action today)\n"
    "## 📋 Categories\n"
    "Group emails into: Job/Placements, Finance, Social, Promotions, Security, Updates, Newsletters\n"
    "For each category, list 1-2 line summaries.\n"
    "## ✅ Action Checklist\n"
    "End with exactly 3 critical action items the user must do.\n\n"
    "Be concise. Use emoji. Total response under 400 words."
)

def run_daily_digests():
    """Fires once a day: for every stored user, fetch recent emails and save an AI digest."""
    print(f"[scheduler] running daily digest job at {datetime.now(DIGEST_TIMEZONE).isoformat()} IST")
    conn = get_db()
    users = conn.execute("SELECT email, credentials_json FROM users").fetchall()
    conn.close()

    for row in users:
        email = row["email"]
        try:
            creds_dict = json.loads(row["credentials_json"])
            creds = Credentials(**creds_dict)
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                update_user_token_only(email, credentials_to_dict(creds))

            service = build("gmail", "v1", credentials=creds)
            results = service.users().messages().list(userId="me", maxResults=30).execute()
            messages = results.get("messages", [])
            if not messages:
                continue

            ids = [m["id"] for m in messages]
            details, _errors = fetch_messages_batch(service, ids, fmt="metadata", metadata_headers=EMAIL_METADATA_HEADERS)
            emails = [build_email_obj(mid, details[mid]) for mid in ids if details.get(mid)]
            if not emails:
                continue

            emails_text = "\n\n---\n\n".join(
                f"From: {e['from']}\nSubject: {e['subject']}\nSnippet: {e['snippet']}" for e in emails
            )
            result = groq_call(DIGEST_SYSTEM_PROMPT, emails_text, max_tokens=900)
            digest_text = result["choices"][0]["message"]["content"]

            save_digest(email, digest_text, len(emails))
            print(f"[scheduler] digest saved for {email} ({len(emails)} emails)")
        except Exception as e:
            # Don't let one user's failure (revoked token, rate limit, etc.) kill the whole run
            print(f"[scheduler] failed for {email}: {e}")

# Guard against APScheduler starting twice under Flask's debug reloader
# (which spawns a watcher process + a worker process).
if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_daily_digests,
        CronTrigger(hour=DIGEST_HOUR, minute=DIGEST_MINUTE, timezone=DIGEST_TIMEZONE),
        id="daily_digest_job",
        replace_existing=True,
    )
    scheduler.start()
    print(f"[scheduler] daily digest scheduled for {DIGEST_HOUR:02d}:{DIGEST_MINUTE:02d} IST every day")

if __name__ == '__main__':
    app.run(debug=True, port=5000)