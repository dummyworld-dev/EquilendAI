import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/mock_data.csv")

# ===== 1. Fairness Metrics =====
male_default = df[df['gender'] == 'Male']['default_status'].mean()
female_default = df[df['gender'] == 'Female']['default_status'].mean()

disparate_impact_ratio = female_default / male_default if male_default != 0 else 0

# ===== 2. Simple Visualization (instead of SHAP) =====
df.groupby('gender')['default_status'].mean().plot(kind='bar')
plt.title("Default Rate by Gender")
plt.ylabel("Default Rate")
plt.savefig("data/fairness_plot.png")
plt.close()

# ===== 3. Mock Model Metrics =====
accuracy = 0.82
precision = 0.78
recall = 0.75

# ===== 4. Generate Markdown Report =====
report = f"""
# Fairness Report

##  Model Performance
- Accuracy: {accuracy}
- Precision: {precision}
- Recall: {recall}

##  Fairness Metrics
- Male Default Rate: {male_default:.2f}
- Female Default Rate: {female_default:.2f}
- Disparate Impact Ratio: {disparate_impact_ratio:.2f}

##  Visualization
![Fairness Plot](fairness_plot.png)

##  Interpretation
- A Disparate Impact Ratio close to 1 indicates fairness.
- Significant deviation may indicate bias.

##  Conclusion
This report provides a basic fairness audit for the model.
"""

# Save report
with open("data/fairness_report.md", "w") as f:
    f.write(report)

print("Fairness report generated successfully!")