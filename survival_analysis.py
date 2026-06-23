from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# lifelines occasionally throws convergence warnings; silence for cleaner output
warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "output" / "data" / "cleaned_telco.csv"
PLOT_DIR  = BASE_DIR / "output" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# --- Build the modelling frame ---

covariates_to_encode = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling"
]


df_surv = pd.get_dummies(df, columns=covariates_to_encode, drop_first=True)


bool_cols = df_surv.select_dtypes(include="bool").columns
df_surv[bool_cols] = df_surv[bool_cols].astype(int)
df_surv = df_surv.select_dtypes(exclude=['object'])
df_surv = df_surv.fillna(df_surv.median())


assert "tenure" in df_surv.columns and "Churn" in df_surv.columns

# --- Fit Cox PH ---
cph = CoxPHFitter()
cph.fit(
    df_surv,
    duration_col="tenure",
    event_col="Churn",
    show_progress=False,
)
print("\n=== Cox PH Summary (top 10 by |coefficient|) ===")
summary = cph.summary
summary["abs_coef"] = summary["coef"].abs()
print(summary.sort_values("abs_coef", ascending=False).head(10).drop(columns="abs_coef"))

# --- Plot survival curves for the three contract types ---

from lifelines import KaplanMeierFitter

fig, ax = plt.subplots(figsize=(9, 6))
kmf = KaplanMeierFitter()
colors = {"Month-to-month": "#d62728",
          "One year":       "#1f77b4",
          "Two year":       "#2ca02c"}

# Re-load the cleaned DF so we have the original Contract strings
df_raw = pd.read_csv(DATA_PATH)

for label, color in colors.items():
    mask = df_raw["Contract"] == label
    kmf.fit(
        durations=df_raw.loc[mask, "tenure"],
        event_observed=df_raw.loc[mask, "Churn"],
        label=label,
    )
    kmf.plot_survival_function(ax=ax, color=color, ci_show=True)

ax.set_title("Survival Curves by Contract Type", fontsize=14, fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Survival Probability  (P still active)")
ax.set_ylim(0, 1.01)
ax.legend(title="Contract Type")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "survival_by_contract.png", dpi=150)
plt.close()
print(f"\nSurvival curve saved to: {PLOT_DIR / 'survival_by_contract.png'}")

# --- Top 5 risk factors (largest positive hazard ratios) ---
top5 = cph.summary.sort_values("coef", ascending=False).head(5)
print("\n=== Top 5 risk factors (largest positive coefficients) ===")
print(top5[["coef", "exp(coef)", "p"]])