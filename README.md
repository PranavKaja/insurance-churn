# Auto Insurance Churn

> Predicting policyholder churn on a badly imbalanced dataset (~11.5% churn), with a data-leakage fix and a SMOTE-rebalanced training pipeline. Five models compared on ROC-AUC; the Gradient Boosting champion is threshold-tuned for recall.

**Result:** Gradient Boosting champion at ROC-AUC ~0.70 on honest features, threshold-tuned to catch the churners that matter.
**Stack:** Python, scikit-learn, imbalanced-learn (SMOTE), pandas
**Live case study:** https://pranavkaja.vercel.app/projects/auto-insurance-churn

## The problem

~92,849 policyholders, but only 11.5% churn. With imbalance that severe a model can hit ~88% accuracy by predicting "nobody churns" and be useless. The goal is to actually catch churners.

## Approach

1. **Leakage fix (the big one).** Dropped `acct_suspd_date`, which is only populated *after* an account is suspended, so it leaks the outcome. An earlier run looked great purely because it was reading the answer. Raw IDs and geo columns were dropped too.
2. **Feature engineering.** Tenure in years, premium-per-tenure, age bands, and a financial-stability score (home owner + good credit + college degree).
3. **SMOTE.** Rebalanced the *training* set only, never the test set, so the models see enough churn examples to learn the minority class without leaking into evaluation.
4. **Model comparison.** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Naive Bayes, ranked by ROC-AUC (the honest metric under imbalance).
5. **Threshold tuning.** Lowered the champion's decision threshold to raise recall, since missing a churner costs more than a false alarm.

## Results

- Gradient Boosting led on ROC-AUC (~0.70 on honest features).
- Removing the leakage column mattered more than the model choice. That's the real takeaway.

Run the script to reproduce the full comparison table and the tuned threshold.

## Run it

```bash
git clone https://github.com/PranavKaja/insurance-churn.git
cd insurance-churn
pip install -r requirements.txt
# place churn_sample.csv (public "Auto Insurance Churn" dataset) in this folder
python insurance_churn_prediction.py
```

## Notes

Public, synthetic "Auto Insurance Churn" dataset (~92,849 policyholders; faker-generated names and addresses, not real people). The data files aren't committed (the address file is ~119 MB and the addresses look like PII even though they're synthetic), so download the dataset and drop `churn_sample.csv` in the repo root.

---
Part of my [portfolio](https://pranavkaja.vercel.app). Built by Pranav Kaja.
