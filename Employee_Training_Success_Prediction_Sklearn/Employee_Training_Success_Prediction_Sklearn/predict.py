from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
model = joblib.load(BASE / "employee_training_success_model.pkl")

sample = pd.DataFrame([{
    "age": 29,
    "years_experience": 4,
    "training_hours": 45,
    "attendance_percent": 92,
    "pre_training_score": 72,
    "department": "IT"
}])

prediction = model.predict(sample)[0]
confidence = max(model.predict_proba(sample)[0]) * 100

print(f"Predicted Training Result: {prediction}")
print(f"Model Confidence: {confidence:.2f}%")
