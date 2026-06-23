from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
OUT_DIR = BASE_DIR / "output" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["customerID"])

# ---------- 2. Fix TotalCharges ----------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
median_total = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

# ---------- 3. Encode all Yes/No binary columns to 1/0 ----------
binary_cols = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "Churn", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

yes_no_map = {"Yes": 1, "No": 0}
for col in binary_cols:
    df[col] = df[col].map(yes_no_map)

# ---------- 4. Save the "cleaned" version (before one-hot) for survival analysis ----------
cleaned_path = OUT_DIR / "cleaned_telco.csv"
df.to_csv(cleaned_path, index=False)
print(f"Cleaned (pre-encoded) data saved to: {cleaned_path}")

# ---------- 5. One-hot encode the specified categoricals ----------
ohe_cols = [
    "InternetService", "Contract", "PaymentMethod",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]
df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

# Convert the dummy columns from bool to int for downstream libraries that prefer it
bool_cols = df_encoded.select_dtypes(include="bool").columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)


y = df_encoded["Churn"]
X = df_encoded.drop(columns=["Churn"])

# ---------- 7. Stratified 80/20 split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Churn rate train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
print(f"Processed files saved to: {OUT_DIR}")