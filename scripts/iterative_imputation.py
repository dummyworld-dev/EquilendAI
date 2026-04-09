from sklearn.experimental import enable_iterative_imputer  # required
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/mock_data.csv")

# Convert gender to numeric
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# ===== 1. Introduce missing values (for demo) =====
df.loc[df.sample(frac=0.1).index, 'monthly_income'] = np.nan
df.loc[df.sample(frac=0.1).index, 'utility_bill_average'] = np.nan

# ===== 2. Apply Iterative Imputer =====
imputer = IterativeImputer(max_iter=10, random_state=0)

imputed_data = imputer.fit_transform(df)

# Convert back to DataFrame
df_imputed = pd.DataFrame(imputed_data, columns=df.columns)

# ===== 3. Save result =====
df_imputed.to_csv("data/imputed_data.csv", index=False)

print("Missing values handled using Iterative Imputer!")