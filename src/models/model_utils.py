# src/models/model_utils.py
import os
import joblib
import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline

# ✅ Import functions from your source files
from train_rf import train_rf_final
from train_xgb import train_xgb_tuned 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_pipeline_from_source(X_train, y_train, numeric_features, categorical_features, model_type="xgb"):
    """
    Builds and trains the final pipeline using the 0.878 accuracy logic.
    """
    if X_train is None or y_train is None or X_train.empty:
        logger.warning("⚠️ No data provided for training. Performing No-Op.")
        return None

    logger.info(f"🚀 Starting pipeline build for model type: {model_type}")
    
    # 1. Preprocessing 
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ]
    )
    
    # 2. Fit and Transform data for the specific model functions
    logger.info("Step 1: Transforming training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Recover feature names for XGBoost/RF compatibility
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_features = list(numeric_features) + list(cat_names)
    X_train_df = pd.DataFrame(X_train_processed, columns=all_features)
    
    # 3. Model Training logic
    logger.info(f"Step 2: Training {model_type.upper()}...")
    
    if model_type == "xgb":
        # Internal split for Early Stopping (required for 0.878 accuracy logic)
        X_t, X_v, y_t, y_v = train_test_split(X_train_df, y_train, test_size=0.1, random_state=42)
        model = train_xgb_tuned(X_t, y_t, X_v, y_v)
        
    elif model_type == "rf":
        model = train_rf_final(X_train_df, y_train)
        
    else:
        logger.error(f"❌ No operation defined for model_type: '{model_type}'")
        return None

    logger.info(f"✅ {model_type.upper()} training complete.")

    # 4. Final Pipeline Construction
    final_pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    return final_pipeline

def save_pipeline(pipeline, path="model/final_pipeline.pkl"):
    if pipeline is None: return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    logger.info(f"💾 Pipeline saved to: {path}")

def load_pipeline(path="model/final_pipeline.pkl"):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ==============================================================================
# EXECUTION BLOCK (This makes it run when you call 'python model_utils.py')
# ==============================================================================
if __name__ == "__main__":
    DATA_PATH = r"D:\EquilendAI\scripts\data\equilend_mock_data.csv"
    
    if os.path.exists(DATA_PATH):
        # 1. Load and Engineer Features
        df = pd.read_csv(DATA_PATH)
        df["bill_income_ratio"] = df["utility_bill_average"] / df["monthly_income"]
        df["risk_proxy"] = (df["utility_bill_average"] / df["monthly_income"]) - (df["repayment_history_pct"] / 100)
        df["income_to_bill_ratio"] = df["monthly_income"] / df["utility_bill_average"]
        df["repayment_bill_interaction"] = df["repayment_history_pct"] * df["utility_bill_average"]
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = pd.get_dummies(df, drop_first=True)
        df.columns = df.columns.str.replace(r"[<>\[\]]", "", regex=True)

        # 2. Setup Train/Test
        X = df.drop("default_status", axis=1)
        y = df["default_status"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Simple Imputation
        medians = X_train.median(numeric_only=True)
        X_train, X_test = X_train.fillna(medians), X_test.fillna(medians)

        # 4. Identify Feature Types
        num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = X_train.select_dtypes(include=['object', 'bool']).columns.tolist()

        # 5. Build and Test
        pipeline = build_pipeline_from_source(X_train, y_train, num_cols, cat_cols, "xgb")
        
        if pipeline:
            y_pred = pipeline.predict(X_test)
            logger.info(f"✨ TEST SET ACCURACY: {accuracy_score(y_test, y_pred):.4f}")
            save_pipeline(pipeline)
    else:
        logger.error(f"Could not find data at {DATA_PATH}")