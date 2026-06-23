# Auto Insurance Churn

> Churn modeling on 92,849 policyholders with severe class imbalance, plus a data-leakage fix that corrected an earlier near-random result.

**Result:** Gradient Boosting champion at ROC-AUC 0.70, threshold-tuned for recall.
**Stack:** Python, scikit-learn, SMOTE, gradient boosting, pandas
**Live case study:** https://pranavkaja.vercel.app/projects/auto-insurance-churn

## The problem

Predict which of 92,849 policyholders will churn when only 11.5% do, so the signal is easy to drown.

## Approach

- Compared five classifiers on a SMOTE pipeline to handle the 11.5% imbalance.
- Found and removed a leakage column that was only set after churn, it had crowned a near-random model in an earlier run.
- Threshold-tuned the champion for recall, since missing a churner is the costly error.

## Results

- Gradient Boosting champion at ROC-AUC 0.70, on honest features.
- A concrete reminder that leakage detection beats model choice.

## Run it

```bash
git clone https://github.com/PranavKaja/insurance-churn.git
cd insurance-churn
pip install -r requirements.txt
jupyter notebook
```

## Notes

Public/anonymized policyholder dataset (92,849 records). No real customer PII.

---
Part of my [portfolio](https://pranavkaja.vercel.app). Built by Pranav Kaja.
