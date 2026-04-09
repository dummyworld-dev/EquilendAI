import os
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

load_dotenv()


def load_data_from_mongo(
    db_name: str = "equilend",
    collection_name: str = "users",
    batch_size: int = 100
) -> pd.DataFrame:
    """
    Securely load data from MongoDB Atlas using environment variables.
    Supports pagination and handles errors gracefully.
    """

    try:
        # Load MongoDB URI from environment variable
        uri = os.getenv("MONGO_URI")

        if not uri:
            raise ValueError("MONGO_URI not found in environment variables")

        # Connect with timeout
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)

        # Force connection check
        client.server_info()

        db = client[db_name]
        collection = db[collection_name]

        # Pagination using batch size
        cursor = collection.find().batch_size(batch_size)

        data = []
        for doc in cursor:
            data.append(doc)

        if not data:
            print(" No data found in collection")
            return pd.DataFrame()

        # Convert BSON → DataFrame
        df = pd.DataFrame(data)

        # Remove MongoDB internal ID
        if "_id" in df.columns:
            df.drop(columns=["_id"], inplace=True)

        print("Data loaded successfully from MongoDB")

        return df

    except Exception as e:
        print("MongoDB not available, using mock data for testing...")

        #  Mock fallback (important for demo/testing)
        mock_data = [
            {"income": 50000, "expenses": 20000, "savings": 10000},
            {"income": 60000, "expenses": 25000, "savings": 15000},
            {"income": 45000, "expenses": 18000, "savings": 8000},
        ]

        return pd.DataFrame(mock_data)


#  Test execution
if __name__ == "__main__":
    df = load_data_forom_mongo()

    print("\n Sample Data:")
    print(df.head())