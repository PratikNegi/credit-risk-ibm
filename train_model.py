import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────
df = pd.read_csv("train.csv")

# ──────────────────────────────────────────────
# 2. Drop non-predictive columns
# ──────────────────────────────────────────────
df.drop(columns=["customer_id", "name"], inplace=True)

# ──────────────────────────────────────────────
# 3. Clean / fill missing values
# ──────────────────────────────────────────────
df["owns_car"] = df["owns_car"].replace("", "N")
df["migrant_worker"] = df["migrant_worker"].replace("", "0.0")
df["gender"] = df["gender"].replace("XNA", "F")  # single XNA row → majority class

df["migrant_worker"] = df["migrant_worker"].astype(float)

# ──────────────────────────────────────────────
# 4. Encode categoricals
# ──────────────────────────────────────────────
label_encoders = {}
for col in ["gender", "owns_car", "owns_house", "occupation_type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ──────────────────────────────────────────────
# 5. Features / target split
# ──────────────────────────────────────────────
TARGET = "credit_card_default"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ──────────────────────────────────────────────
# 6. Train model
# ──────────────────────────────────────────────
print("Training Random Forest …")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ──────────────────────────────────────────────
# 7. Evaluate
# ──────────────────────────────────────────────
y_pred = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
roc = roc_auc_score(y_val, y_proba)
report = classification_report(y_val, y_pred, output_dict=True)
cm = confusion_matrix(y_val, y_pred).tolist()

print(f"Validation Accuracy : {acc:.4f}")
print(f"ROC-AUC             : {roc:.4f}")
print(classification_report(y_val, y_pred))

# ──────────────────────────────────────────────
# 8. Feature importance
# ──────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# ──────────────────────────────────────────────
# 9. Save artifacts
# ──────────────────────────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

metrics = {
    "accuracy": round(acc, 4),
    "roc_auc": round(roc, 4),
    "classification_report": report,
    "confusion_matrix": cm,
    "feature_names": FEATURES,
    "feature_importances": importance_df.to_dict(orient="records"),
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved: model.pkl | label_encoders.pkl | metrics.json")
