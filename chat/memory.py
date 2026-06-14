from langchain_core.chat_history import InMemoryChatMessageHistory

_sessions: dict[str, InMemoryChatMessageHistory] = {}

def get_or_create_memory(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _sessions:
        _sessions[session_id] = InMemoryChatMessageHistory()
    return _sessions[session_id]

def clear_memory(session_id: str) -> None:
    if session_id in _sessions:
        _sessions[session_id].clear()

def delete_session(session_id: str) -> None:
    _sessions.pop(session_id, None)

def list_sessions() -> list[str]:
    return list(_sessions.keys())