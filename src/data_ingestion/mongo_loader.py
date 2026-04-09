"""
Task 01 - MongoDB Data Ingestion

Two function groups:
  load_data_from_mongo()  - pulls raw training data FROM MongoDB (Task 01a)
  save_decision()         - persists a credit decision TO MongoDB  (Task 01b)
  load_decisions()        - retrieves saved decisions FROM MongoDB (Task 01b)
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False


def _load_env_file(path: str = ".env") -> None:
    """Load local .env values without requiring python-dotenv."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


# -- Internal helper ----------------------------------------------------------

def _get_client(timeout_ms: int = 5000):
    """Return a connected MongoClient, or None if unavailable."""
    if not _PYMONGO_AVAILABLE:
        return None
    uri = os.getenv("MONGO_URI") or os.getenv("MONGODB_URI")
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.server_info()
        return client
    except Exception:
        return None


# -- Task 01a: load raw dataset from MongoDB ----------------------------------

def load_data_from_mongo(
    db_name: str = "equilend",
    collection_name: str = "users",
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Securely load data from MongoDB Atlas using environment variables.
    Supports pagination and falls back to a mock DataFrame on failure.
    """
    try:
        client = _get_client()
        if client is None:
            raise ValueError("MONGO_URI not found in environment variables")

        collection = client[db_name][collection_name]
        cursor = collection.find().batch_size(batch_size)
        data = list(cursor)

        if not data:
            print("No data found in collection.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        print("Data loaded successfully from MongoDB.")
        return df

    except Exception as e:
        print(f"MongoDB not available ({e}), using mock data for testing...")
        mock_data = [
            {"income": 50000, "expenses": 20000, "savings": 10000},
            {"income": 60000, "expenses": 25000, "savings": 15000},
            {"income": 45000, "expenses": 18000, "savings": 8000},
        ]
        return pd.DataFrame(mock_data)


# -- Task 01b: persist / retrieve credit decisions ----------------------------

def save_decision(
    record: dict,
    db: str = "equilend",
    collection: str = "decisions",
) -> bool:
    """
    Persist a credit decision document to MongoDB.
    Raises RuntimeError if MongoDB is unavailable.
    """
    client = _get_client()
    if client is None:
        raise RuntimeError("MongoDB not configured or unreachable.")

    record = dict(record)
    record["saved_at"] = datetime.now(timezone.utc).isoformat()
    client[db][collection].insert_one(record)
    return True


def load_decisions(
    db: str = "equilend",
    collection: str = "decisions",
) -> list:
    """
    Retrieve all saved credit decisions. Returns empty list if unavailable.
    """
    client = _get_client()
    if client is None:
        return []
    return list(client[db][collection].find({}, {"_id": 0}))


# -- CLI test -----------------------------------------------------------------

if __name__ == "__main__":
    df = load_data_from_mongo()
    print("\nSample Data:")
    print(df.head())
