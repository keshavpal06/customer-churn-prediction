from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "output" / "data"
MODEL_DIR = BASE_DIR / "output" / "models"
PLOT_DIR = BASE_DIR / "output" / "plots"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Load processed data ----------
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test  = pd.read_csv(DATA_DIR / "X_test.csv")
X_train = X_train.select_dtypes(exclude=['object'])
X_test  = X_test.select_dtypes(exclude=['object'])
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test  = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# ---------- Compute scale_pos_weight ----------

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

# ---------- Train ----------
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---------- Save model ----------
model_path = MODEL_DIR / "churn_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to: {model_path}")

# ---------- Predict probabilities ----------
y_proba = model.predict_proba(X_test)[:, 1]
y_pred_05 = (y_proba >= 0.5).astype(int)

# ---------- Evaluation ----------
print("\n=== Classification report @ threshold=0.5 ===")
print(classification_report(y_test, y_pred_05, target_names=["No Churn", "Churn"]))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc_auc:.4f}")

# Confusion matrix at 0.5
cm = confusion_matrix(y_test, y_pred_05)
print("Confusion matrix @0.5:\n", cm)

# Save ROC curve
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve — XGBoost")
plt.tight_layout()
plt.savefig(PLOT_DIR / "roc_curve.png", dpi=150)
plt.close()

# Save Precision-Recall curve
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision-Recall Curve — XGBoost")
plt.tight_layout()
plt.savefig(PLOT_DIR / "pr_curve.png", dpi=150)
plt.close()

# Save confusion matrix heatmap at 0.5
fig, ax = plt.subplots()
sns = __import__("seaborn")  # local import to keep imports tidy
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix @ 0.5")
plt.tight_layout()
plt.savefig(PLOT_DIR / "confusion_matrix_05.png", dpi=150)
plt.close()

print("ROC, PR and confusion-matrix plots saved to output/plots/")