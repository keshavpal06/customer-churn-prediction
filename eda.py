from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
PLOT_DIR = BASE_DIR / "output" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load dataset ---
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("Churn value counts:\n", df["Churn"].value_counts())

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
PALETTE = sns.color_palette("Set2")

# ---------- 1. Churn distribution (pie) ----------
fig, ax = plt.subplots()
counts = df["Churn"].value_counts()
ax.pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=PALETTE,
    startangle=90,
    wedgeprops={"edgecolor": "white"},
)
ax.set_title("Churn Distribution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / "churn_distribution.png", dpi=150)
plt.close()

# ---------- 2. Churn rate by Contract type (bar) ----------
fig, ax = plt.subplots()
contract_churn = (
    df.groupby("Contract")["Churn"]
    .value_counts(normalize=True)
    .rename("rate")
    .reset_index()
)
sns.barplot(
    data=contract_churn[contract_churn["Churn"] == "Yes"],
    x="Contract", y="rate", palette="Set2", ax=ax
)
ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
ax.set_ylabel("Churn Rate")
ax.set_xlabel("Contract Type")
# Add labels on top of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "churn_by_contract.png", dpi=150)
plt.close()

# ---------- 3. Tenure distribution by churn status (histogram) ----------
fig, ax = plt.subplots()
sns.histplot(
    data=df, x="tenure", hue="Churn", multiple="stack",
    bins=30, palette="Set2", ax=ax
)
ax.set_title("Tenure Distribution by Churn Status", fontsize=14, fontweight="bold")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(PLOT_DIR / "tenure_distribution.png", dpi=150)
plt.close()

# ---------- 4. Monthly Charges vs Churn (boxplot) ----------
fig, ax = plt.subplots()
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
sns.boxplot(
    data=df, x="Churn", y="MonthlyCharges", palette="Set2", ax=ax
)
ax.set_title("Monthly Charges vs Churn", fontsize=14, fontweight="bold")
ax.set_xlabel("Churn")
ax.set_ylabel("Monthly Charges ($)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "monthly_charges_boxplot.png", dpi=150)
plt.close()

print(f"\nAll EDA plots saved to: {PLOT_DIR}")
for p in PLOT_DIR.glob("*.png"):
    print("  -", p.name)