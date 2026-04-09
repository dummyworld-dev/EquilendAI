import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

try:
    from pymongo import MongoClient
    from pymongo.collection import Collection
except ModuleNotFoundError:
    MongoClient = None
    Collection = Any


def _mongo_uri() -> Optional[str]:
    """
    Resolve Mongo URI from environment.
    """
    return os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")


def get_collection(
    db_name: str = "equilend_ai",
    collection_name: str = "loan_decisions",
) -> Optional[Collection]:
    """
    Return Mongo collection handle if URI is configured; otherwise None.
    """
    uri = _mongo_uri()
    if not uri or MongoClient is None:
        return None

    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    return client[db_name][collection_name]


def save_decision(
    payload: Dict[str, Any],
    db_name: str = "equilend_ai",
    collection_name: str = "loan_decisions",
) -> bool:
    """
    Persist a decision document to MongoDB. Returns True on success.
    """
    collection = get_collection(db_name=db_name, collection_name=collection_name)
    if collection is None:
        return False

    doc = dict(payload)
    doc["created_at"] = datetime.now(timezone.utc).isoformat()
    collection.insert_one(doc)
    return True


def fetch_recent_decisions(
    limit: int = 20,
    db_name: str = "equilend_ai",
    collection_name: str = "loan_decisions",
) -> List[Dict[str, Any]]:
    """
    Fetch recent decision records, newest first.
    """
    collection = get_collection(db_name=db_name, collection_name=collection_name)
    if collection is None:
        return []

    cursor = collection.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    return list(cursor)

