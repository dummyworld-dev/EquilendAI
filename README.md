```markdown
#  EquiLend AI: Open Source ML Challenge 2026

##  The Problem
Traditional credit scoring relies heavily on historical banking data, excluding millions of "credit invisible" individuals. **EquiLend AI** aims to bridge this gap by using alternative data—such as utility payments, cash flow consistency, and digital footprints—to assess risk fairly and transparently.

##  The Challenge
As a participant, you are provided with a **Streamlit UI Prototype**. However, the "brain" of the application is currently a simple mathematical placeholder with several **critical logical flaws**. 

### Your Mission:
1.  **Fix the Logic:** Identify and resolve the 4 logical errors in the `app.py` starter code.
2.  **Build the ML Pipeline:** Transition from a basic formula to a robust **XGBoost/LightGBM** model.
3.  **Ensure Fairness:** Implement bias detection (Disparate Impact Ratio) to ensure the model doesn't discriminate.
4.  **Persistence:** Connect the dashboard to **MongoDB Atlas** to ensure every loan decision is audited and saved.

---

##  Tech Stack
* **Language:** Python 3.10+
* **Frontend:** Streamlit
* **ML Libraries:** Scikit-learn, XGBoost, SHAP (for explainability)
* **Database:** MongoDB Atlas (NoSQL)
* **Validation:** Pydantic

---

##  Repository Structure
* `app.py`: The main Streamlit dashboard (Contains intentional bugs).
* `model_utils.py`: Placeholder for your training and inference logic.
* `requirements.txt`: Necessary Python dependencies.
* `.env.example`: Template for your MongoDB connection string.
* `data/`: Folder for datasets (e.g., `sample_loans.csv`).

---

##  Getting Started

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed. We recommend using a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment
Copy the example environment file and add your **MongoDB Atlas URI**:
```bash
cp .env.example .env
```

### 4. Run the Dashboard
If the `streamlit` command is not in your PATH, use the python module execution:
```bash
python -m streamlit run app.py
```

---

##  The "Answer Key": Hidden Logical Bugs
To qualify for a top-tier score, your submission **must** address these 4 hidden errors found in the starter kit:

* **Division by Zero:** The current score logic fails if a `utility_bill` is entered as 0.
* **Age Guard Bypass:** The system currently allows scoring for minors (Age < 18).
* **Linear Scaling Flaw:** The current formula is a simple ratio. Real-world risk is non-linear and requires a trained ML model.
* **State Persistence:** Currently, clicking "Analyze" displays a result that disappears on refresh. You must implement a database save to MongoDB.

---

##  Evaluation Rubrics
| **Model Quality** | Optimized XGBoost with AUC > 0.85. | Basic Random Forest. | Hard-coded logic. |
| **Explainability** | Interactive SHAP plots for every loan. | Static feature importance. | No explainability. |
| **Fairness** | Bias detection script & metrics included. | Mention of fairness in README. | No bias checking. |
| **Security** | Pydantic validation & `.env` usage. | Basic `try-except` blocks. | Hard-coded credentials. |

---

## Submission Guidelines
1. Fork this repository.
2. Complete the **Set 1 (ML Engine)** and **Set 2 (Dashboard)** tasks.
3. Include a `Fairness_Report.md` summarizing your model's performance.
```

---
