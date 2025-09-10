from pymongo import MongoClient
import os

from dotenv import load_dotenv
load_dotenv()


client = None
db = None
users_collection = None
sessions_collection = None
messages_collection = None


def init_connection():
    """Initialize MongoDB connection and collections."""
    global client, db, users_collection, sessions_collection, messages_collection

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI not found in environment variables.")

    client = MongoClient(mongo_uri)
    db = client.get_database("law_cases_db")

    users_collection = db.get_collection("users")
    sessions_collection = db.get_collection("chat_sessions")
    messages_collection = db.get_collection("chat_messages")

    # ✅ create unique index (username + normalized chat name)
    sessions_collection.create_index(
        [("username", 1), ("session_name_normalized", 1)],
        unique=True
    )

    return {
        "client": client,
        "db": db,
        "users": users_collection,
        "sessions": sessions_collection,
        "messages": messages_collection
    }
