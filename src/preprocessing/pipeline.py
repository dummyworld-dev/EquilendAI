"""
Task 04 — Scaling & Encoding Pipeline
Builds a scikit-learn ColumnTransformer that handles both numeric
(imputation + scaling) and categorical (imputation + one-hot) features.
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer

NUMERIC_FEATURES     = ["monthly_income", "utility_bill_average", "repayment_history_pct"]
CATEGORICAL_FEATURES = ["employment_length", "gender"]


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer.

    Numeric path  : IterativeImputer (Task 03) → StandardScaler
    Categorical path: SimpleImputer (most_frequent) → OneHotEncoder

    Returns:
        sklearn.compose.ColumnTransformer ready to fit_transform / transform.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", IterativeImputer(max_iter=10, random_state=42)),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer,     NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor
