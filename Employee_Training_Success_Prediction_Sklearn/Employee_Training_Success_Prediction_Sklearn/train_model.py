from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "data" / "employee_training_success.csv")

X = df.drop(columns=["training_result"])
y = df["training_result"]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), ["department"]),
    ("numeric", "passthrough", [
        "age", "years_experience", "training_hours",
        "attendance_percent", "pre_training_score"
    ])
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred))

joblib.dump(model, BASE / "employee_training_success_model.pkl")

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.title("Employee Training Success Prediction")
plt.tight_layout()
plt.savefig(BASE / "confusion_matrix.png", dpi=150)
print("Model and confusion matrix saved successfully.")
