import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def generate_shap_explanation(model, X_instance, output_dir="src/evaluation"):
    """
    Generates a SHAP waterfall plot to explain a specific loan decision.
    X_instance: A single row DataFrame representing one applicant.
    """
    # 1. Initialize the TreeExplainer (Optimized for Random Forest/XGBoost)
    explainer = shap.TreeExplainer(model)
    
    # 2. Calculate SHAP values
    # check_additivity=False helps if there are minor float discrepancies in tree models
    shap_values = explainer(X_instance, check_additivity=False)
    
    # 3. Visualization
    plt.figure(figsize=(10, 6))
    
    # We explain index [1] which is 'Default' probability
    # If shap_values has 3 dimensions (instance, feature, class), we slice it
    if len(shap_values.shape) == 3:
        instance_explanation = shap_values[0, :, 1]
    else:
        instance_explanation = shap_values[0]

    shap.plots.waterfall(instance_explanation, max_display=10, show=False)
    
    plt.title("EquiLend AI: Loan Decision Factor Analysis", fontsize=14, pad=20)
    
    # 4. Save with absolute path logic
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "shap_waterfall.png")
    
    # bbox_inches='tight' prevents labels from being cut off
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return save_path
