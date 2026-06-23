# Auto Insurance Churn

> Churn modeling on ~92,849 policyholders with an 11.5% churn rate, plus a data-leakage fix that removed a column only set after a customer had already churned.

**Result:** Five classifiers compared with class-weight balancing; Gradient Boosting led on ROC-AUC (~0.70 on honest features).
**Stack:** Python, scikit-learn, pandas
**Live case study:** https://pranavkaja.vercel.app/projects/auto-insurance-churn

## The problem

Predict which of ~92,849 policyholders will churn when only 11.5% do, so the signal is easy to drown.

## Approach

- Compared five classifiers (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Naive Bayes), using `class_weight='balanced'` to handle the 11.5% imbalance.
- Dropped a leakage column (`acct_suspd_date`, set only after an account is suspended) along with raw IDs and geo fields, so the model learns from honest features. An earlier run looked great only because that column was leaking the answer.
- Engineered tenure, premium-per-tenure, age bands, and a financial-stability score.
- Reported ROC-AUC, precision, recall, and F1 per model. ROC-AUC is the headline metric here, since missing a churner is the costly error.

## Results

- Gradient Boosting was the strongest on ROC-AUC (~0.70 on honest features).
- The bigger lesson: removing the leakage column mattered more than the model choice.

## Run it

```bash
git clone https://github.com/PranavKaja/insurance-churn.git
cd insurance-churn
pip install -r requirements.txt
# place churn_sample.csv in this folder (see the dataset note below)
python insurance_churn_prediction.py
```

## Notes

Uses the public, synthetic "Auto Insurance Churn" dataset (~92,849 policyholders; faker-generated names and addresses, not real people). The data files aren't committed (they're large and the addresses, while synthetic, look like PII), so grab the public dataset and drop `churn_sample.csv` in the repo root to run.

---
Part of my [portfolio](https://pranavkaja.vercel.app). Built by Pranav Kaja.
