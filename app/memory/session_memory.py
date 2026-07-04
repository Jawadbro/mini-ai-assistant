"""
In-process conversation memory, keyed by session_id.

This is intentionally simple: a dict of session_id -> list of turns,
kept in process memory. It resets on server restart. For a production
deployment this would be swapped for Redis or a database-backed store
behind the same get_history/add_turn interface -- nothing else in the
codebase would need to change.
"""
import threading
from collections import defaultdict
from typing import List, Dict

from app.config import MAX_HISTORY_TURNS

_lock = threading.Lock()
_sessions: Dict[str, List[dict]] = defaultdict(list)


def add_turn(session_id: str, role: str, content: str) -> None:
    with _lock:
        _sessions[session_id].append({"role": role, "content": content})
        # Keep only the most recent N turns to bound prompt size
        if len(_sessions[session_id]) > MAX_HISTORY_TURNS * 2:
            _sessions[session_id] = _sessions[session_id][-MAX_HISTORY_TURNS * 2:]


def get_history(session_id: str) -> List[dict]:
    with _lock:
        return list(_sessions.get(session_id, []))


def clear_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)
