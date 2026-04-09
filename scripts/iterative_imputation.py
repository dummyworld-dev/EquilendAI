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

# ===== 4. Outlier Detection (IQR Method) =====

def count_outliers(df):
    outlier_count = 0

    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        count = len(outliers)

        print(f"Outliers in {col}: {count}")
        outlier_count += count

    print(f"\nTotal Outliers in dataset: {outlier_count}")


# Call function
count_outliers(df_imputed)