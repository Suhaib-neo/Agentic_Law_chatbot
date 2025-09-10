from datetime import datetime
from bson.objectid import ObjectId

def load_user_sessions(username, sessions_collection, last_session_id=None):
    """
    Load sessions for a user. Restore last active session if possible.
    Returns: (sessions_list, current_session_or_None, messages_list)
    """
    if sessions_collection is None:
        return [], None, []

    sessions = list(sessions_collection.find({"username": username}).sort("timestamp", -1))
    current_session = None
    messages = []

    if sessions:
        if last_session_id:
            try:
                last = sessions_collection.find_one({"_id": ObjectId(last_session_id)})
            except Exception:
                last = None
            if last:
                current_session = last
                # messages will be loaded by caller or by calling load_chat_history
        if not current_session:
            current_session = sessions[0]

    # Note: do NOT load messages here unless you also have messages_collection.
    # Return sessions and current_session; caller can call load_chat_history with messages_collection.
    return sessions, current_session, messages


def load_chat_history(session_id, messages_collection):
    """
    Loads messages for a given chat session from messages_collection.
    """
    if messages_collection is None:
        return []
    try:
        msgs = list(messages_collection.find({"session_id": session_id}).sort("timestamp", 1))
        return [{"role": m.get("role", "assistant"), "content": m.get("content", "")} for m in msgs]
    except Exception:
        return []


def save_message(session_id, role, content, messages_collection):
    """
    Save a message to the chat history in messages_collection.
    """
    if messages_collection is None:
        return None
    try:
        doc = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
        }
        return messages_collection.insert_one(doc).inserted_id
    except Exception:
        return None
