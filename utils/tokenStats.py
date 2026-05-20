import json
import os
import threading
import time
from datetime import datetime
from hashlib import sha256

import utils.globals as globals


RECENT_LIMIT = int(os.getenv("TOKEN_USAGE_RECENT_LIMIT", "300"))
_usage_lock = threading.Lock()


def token_id(token):
    return sha256(token.encode()).hexdigest()


def token_kind(token):
    if token.startswith("eyJhbGciOi") or token.startswith("fk-"):
        return "access"
    if len(token) == 45 or token.startswith("rt_"):
        return "refresh"
    return "unknown"


def mask_token(token):
    if len(token) <= 18:
        return token[:4] + "..." if token else ""
    return f"{token[:10]}...{token[-8:]}"


def format_timestamp(timestamp):
    if not timestamp:
        return None
    try:
        return datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def save_token_usage_stats():
    with open(globals.TOKEN_USAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(globals.token_usage_stats, f, ensure_ascii=False, indent=2)


def _empty_summary():
    return {
        "total": 0,
        "success": 0,
        "failure": 0,
        "last_1m": 0,
        "last_5m": 0,
        "last_1h": 0,
        "last_24h": 0,
        "last_used_at": None,
        "last_success_at": None,
        "last_error_at": None,
        "last_error": None,
        "types": {},
        "models": {},
        "recent": [],
    }


def record_token_usage(token, request_type="chat", model=None, success=True, status_code=None, error=None):
    if not token:
        return

    now = int(time.time())
    event = {
        "timestamp": now,
        "time": format_timestamp(now),
        "type": request_type or "chat",
        "model": model or "",
        "success": bool(success),
    }
    if status_code is not None:
        event["status_code"] = status_code
    if error:
        event["error"] = str(error)[:500]

    with _usage_lock:
        stats = globals.token_usage_stats.setdefault(
            token_id(token),
            {
                "id": token_id(token),
                "masked": mask_token(token),
                "type": token_kind(token),
                "total": 0,
                "success": 0,
                "failure": 0,
                "types": {},
                "models": {},
                "recent": [],
            },
        )
        stats["masked"] = mask_token(token)
        stats["type"] = token_kind(token)
        stats["total"] = int(stats.get("total", 0)) + 1
        stats["success" if success else "failure"] = int(stats.get("success" if success else "failure", 0)) + 1
        stats["last_used_at"] = now
        if success:
            stats["last_success_at"] = now
        else:
            stats["last_error_at"] = now
            stats["last_error"] = str(error or status_code or "request failed")[:500]

        request_type = request_type or "chat"
        stats.setdefault("types", {})[request_type] = int(stats.setdefault("types", {}).get(request_type, 0)) + 1
        if model:
            stats.setdefault("models", {})[model] = int(stats.setdefault("models", {}).get(model, 0)) + 1

        recent = stats.setdefault("recent", [])
        recent.append(event)
        if len(recent) > RECENT_LIMIT:
            del recent[:-RECENT_LIMIT]

        save_token_usage_stats()


def summarize_token_usage(token):
    if not token:
        return _empty_summary()
    now = int(time.time())
    with _usage_lock:
        stats = globals.token_usage_stats.get(token_id(token), {})
        if not stats:
            return _empty_summary()
        recent = list(stats.get("recent", []))

    def count_since(seconds):
        threshold = now - seconds
        return len([event for event in recent if int(event.get("timestamp", 0)) >= threshold])

    return {
        "total": int(stats.get("total", 0)),
        "success": int(stats.get("success", 0)),
        "failure": int(stats.get("failure", 0)),
        "last_1m": count_since(60),
        "last_5m": count_since(5 * 60),
        "last_1h": count_since(60 * 60),
        "last_24h": count_since(24 * 60 * 60),
        "last_used_at": format_timestamp(stats.get("last_used_at")),
        "last_success_at": format_timestamp(stats.get("last_success_at")),
        "last_error_at": format_timestamp(stats.get("last_error_at")),
        "last_error": stats.get("last_error"),
        "types": stats.get("types", {}),
        "models": stats.get("models", {}),
        "recent": recent[-20:],
    }


def delete_token_usage(token):
    if not token:
        return
    with _usage_lock:
        globals.token_usage_stats.pop(token_id(token), None)
        save_token_usage_stats()


def reset_token_usage():
    with _usage_lock:
        globals.token_usage_stats.clear()
        save_token_usage_stats()
