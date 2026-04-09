from faker import Faker
import pandas as pd
import random

fake = Faker()

data = []

for _ in range(100):
    gender = random.choice(['Male', 'Female'])
    income = random.randint(20000, 100000)
    utility_bill = random.randint(500, 5000)
    repayment_history = random.randint(50, 100)
    employment_length = random.randint(0, 20)
    default_status = random.choice([0, 1])

    data.append({
        'gender': gender,
        'monthly_income': income,
        'utility_bill_average': utility_bill,
        'repayment_history_pct': repayment_history,
        'employment_length': employment_length,
        'default_status': default_status
    })

df = pd.DataFrame(data)
df.to_csv("data/mock_data.csv", index=False)

print("Mock data generated successfully!")