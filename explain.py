from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "output" / "data"
MODEL_DIR = BASE_DIR / "output" / "models"
PLOT_DIR  = BASE_DIR / "output" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load model and data ---
with open(MODEL_DIR / "churn_model.pkl", "rb") as f:
    model = pickle.load(f)
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
X_test = X_test.select_dtypes(exclude=['object'])
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# --- SHAP Tree explainer ---
explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    shap_values = shap_values[1]
expected_value = (
    explainer.expected_value[1]
    if isinstance(explainer.expected_value, (list, np.ndarray))
    else explainer.expected_value
)

# ---------- 1. Global summary ----------
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    shap_values, X_test, plot_type="bar",
    max_display=15, show=False
)
plt.title("SHAP Global Feature Importance (Top 15)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "shap_summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_summary_bar.png")

# ---------- 2. Local explanation for highest-risk customer ----------
y_proba = model.predict_proba(X_test)[:, 1]
high_risk_idx = int(np.argmax(y_proba))
print(f"Highest-risk customer index: {high_risk_idx}, "
      f"churn probability = {y_proba[high_risk_idx]:.3f}")

# Build a shap.Explanation object so we can call the modern waterfall plotter
explanation = shap.Explanation(
    values=shap_values[high_risk_idx],
    base_values=expected_value,
    data=X_test.iloc[high_risk_idx],
    feature_names=list(X_test.columns),
)

fig, ax = plt.subplots(figsize=(10, 7))
shap.plots.waterfall(explanation, max_display=15, show=False)
plt.title(
    f"SHAP Waterfall — Highest-Risk Customer "
    f"(P(churn) = {y_proba[high_risk_idx]:.2f})",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(PLOT_DIR / "shap_waterfall_high_risk.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: shap_waterfall_high_risk.png")