import os

import pytest

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None


@pytest.mark.skipif(MongoClient is None, reason="pymongo is not installed")
def test_mongodb_connection_env_configured():
    """
    Basic sanity check that a MongoDB URI is configured and a client
    can be instantiated. This does not require the database to be reachable.
    """
    mongo_uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    assert mongo_uri is not None, "MONGODB_URI or MONGO_URI environment variable must be set for MongoDB tests."

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=1000)

    # We only check that server_info can be called without raising a timeout;
    # if the server is unreachable, this test will fail, signaling connectivity issues.
    try:
        client.server_info()
    finally:
        client.close()

