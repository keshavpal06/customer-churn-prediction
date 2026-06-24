# Customer Churn Prediction & Early Warning System

An end-to-end ML system that detects which telecom subscribers 
are likely to leave and when, giving retention teams a chance 
to intervene with targeted incentives before revenue is lost.

## Results
- **ROC-AUC**: 0.8355
- **Business-cost-optimized threshold**: 0.24
- **Churners caught**: 333 out of 374 (89% catch rate)
- **Minimum total retention cost**: $333,500 on test set
- **Annualized revenue at risk identified**: $1,452,475

## What makes this different from a standard classifier
Most churn models optimize for accuracy or F1. This system 
optimizes for **business cost** — assigning asymmetric costs 
to false negatives (missed churners = lost revenue) vs false 
positives (unnecessary retention offers). The threshold of 
0.24 was chosen by minimizing total cost across the test set, 
not by default 0.5.

## Features
- XGBoost classifier with `scale_pos_weight` for class imbalance
- Business-cost threshold optimization (FN=₹3000, FP=₹500)
- SHAP explainability — global feature importance + per-customer
  waterfall explanations
- Cox Proportional Hazards survival analysis — predicts *when* 
  a customer will churn, not just *if*
- Kaplan-Meier survival curves by contract type
- Interactive Streamlit dashboard with 4 tabs

## Tech Stack
Python, XGBoost, SHAP, lifelines, Streamlit, scikit-learn, 
pandas, plotly

## Project Structure
churn_project/

├── data/                        # Telco Customer Churn CSV

├── output/

│   ├── plots/                   # EDA, SHAP, survival plots

│   ├── models/                  # Trained model + threshold

│   └── data/                    # Processed train/test CSVs

├── 1_eda.py                     # Exploratory data analysis

├── 2_preprocessing.py           # Cleaning + encoding + split

├── 3_train_model.py             # XGBoost training + evaluation

├── 4_threshold_optimization.py  # Business cost optimization

├── 5_explain.py                 # SHAP global + local plots

├── 6_survival_analysis.py       # Cox PH + Kaplan-Meier curves

└── 7_app.py                     # Streamlit dashboard

## Setup
```bash
conda create -n churn_env python=3.10
conda activate churn_env
pip install -r requirements.txt
```

## Run pipeline
```bash
python 1_eda.py
python 2_preprocessing.py
python 3_train_model.py
python 4_threshold_optimization.py
python 5_explain.py
python 6_survival_analysis.py
streamlit run 7_app.py
```

## Known limitations & future work
- Single global threshold applied to all customers regardless 
  of their value tier. A high-value customer ($150/month) 
  warrants a lower threshold (more aggressive retention) than 
  a low-value customer ($30/month). Next step: segment-specific 
  thresholds based on customer LTV.
- Survival analysis currently uses Kaplan-Meier for curves and 
  Cox PH for risk factors separately; a unified joint model 
  would improve time-to-churn precision.

## Dataset
Telco Customer Churn — IBM Sample Dataset via Kaggle  
7,043 customers, 21 features, 26.5% churn rate

## Live Demo
[View deployed app](https://keshavpal06-churn-prediction.streamlit.app)