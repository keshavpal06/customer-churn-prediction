from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "output" / "data"
MODEL_DIR = BASE_DIR / "output" / "models"
PLOT_DIR  = BASE_DIR / "output" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- Business costs ---
FN_COST = 3000   # cost of missing a churner (lost customer LTV)
FP_COST = 500     # cost of falsely flagging a loyal customer (retention offer)


with open(MODEL_DIR / "churn_model.pkl", "rb") as f:
    model = pickle.load(f)
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
X_test = X_test.select_dtypes(exclude=['object'])
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# --- Probabilities ---
y_proba = model.predict_proba(X_test)[:, 1]

# --- Sweep thresholds ---
thresholds = np.arange(0.01, 1.00, 0.01)
costs, fns, fps = [], [], []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    # sklearn confusion_matrix: rows = actual, cols = predicted
    # [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fn, fp = cm[1, 0], cm[0, 1]
    costs.append(fn * FN_COST + fp * FP_COST)
    fns.append(fn)
    fps.append(fp)

costs = np.array(costs)
fns   = np.array(fns)
fps   = np.array(fps)

best_idx   = int(np.argmin(costs))
best_t     = float(thresholds[best_idx])
best_cost  = float(costs[best_idx])

print(f"Optimal threshold : {best_t:.2f}")
print(f"Minimum total cost: ${best_cost:,.0f}  "
      f"(FN={fns[best_idx]}, FP={fps[best_idx]})")


thr_path = MODEL_DIR / "threshold.txt"
with open(thr_path, "w") as f:
    f.write(f"{best_t:.4f}\n")
print(f"Threshold saved to: {thr_path}")

# --- Plot threshold vs total cost ---
fig, ax = plt.subplots()
ax.plot(thresholds, costs, color="#1f77b4", linewidth=2, label="Total cost")
ax.axvline(best_t, color="red", linestyle="--",
           label=f"Optimal threshold = {best_t:.2f}")
ax.scatter([best_t], [best_cost], color="red", zorder=5)
ax.set_title("Threshold vs Total Business Cost", fontsize=14, fontweight="bold")
ax.set_xlabel("Probability Threshold")
ax.set_ylabel("Total Cost ($)")
ax.yaxis.set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "threshold_cost_curve.png", dpi=150)
plt.close()
print(f"Cost curve saved to: {PLOT_DIR / 'threshold_cost_curve.png'}")