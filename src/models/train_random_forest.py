import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from src.preprocessing.data_cleaning import load_and_clean
from src.preprocessing.feature_encoding import encode_categorical_features
from src.preprocessing.scaling import scale_numeric_features


def load_prepared_data(csv_path: str) -> pd.DataFrame:
    """
    Load, clean, encode, and scale the EquiLend dataset.

    This helper stitches together the preprocessing steps so model training
    scripts can stay focused on modeling rather than data wrangling.
    """
    # 1. Load and clean raw data (handles missing values, outliers, etc.).
    df = load_and_clean(csv_path)

    # 2. One-hot encode categorical features.
    df = encode_categorical_features(df)

    # 3. Scale numeric features for better model behavior.
    df, _ = scale_numeric_features(df)

    return df


def train_random_forest(
    csv_path: str,
    model_output_path: str = "models/random_forest_model.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Train a Random Forest classifier on the cleaned and preprocessed dataset.

    Includes:
    - Train/test split.
    - Model training.
    - Evaluation metrics printed to stdout.
    - Saving of the trained model to disk.
    """
    # Prepare the data
    df = load_prepared_data(csv_path)

    # Separate features and target
    if "default_status" not in df.columns:
        raise ValueError("Expected 'default_status' column as the target.")

    X = df.drop(columns=["default_status"])
    y = df["default_status"]

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Initialize the Random Forest classifier with sensible defaults.
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )

    # Fit the model
    clf.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Random Forest Evaluation Metrics")
    print("-" * 40)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Save the trained model to disk for later use (e.g., in Streamlit app).
    joblib.dump(clf, model_output_path)
    print(f"\nRandom Forest model saved to: {model_output_path}")


if __name__ == "__main__":
    # Default path assumes the synthetic data generator saved here.
    default_csv_path = "data/equilend_mock_data.csv"
    train_random_forest(default_csv_path)

