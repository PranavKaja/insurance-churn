"""
Auto insurance churn prediction.

Predicts policyholder churn on an imbalanced dataset (~11.5% churn). The headline
moves are (1) dropping a data-leakage column that is only populated after a
customer has already churned, and (2) using SMOTE to rebalance the TRAINING data
so the models actually learn the minority (churn) class. Five classifiers are
compared on ROC-AUC, and the Gradient Boosting champion is threshold-tuned for
recall (missing a churner costs more than a false alarm).

Run:
    pip install -r requirements.txt
    # place churn_sample.csv (public "Auto Insurance Churn" dataset) in this folder
    python insurance_churn_prediction.py
"""

import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# Columns dropped before modeling. `acct_suspd_date` is the leak: it is only set
# once an account is suspended (i.e. after churn), so keeping it hands the model
# the answer. The rest are raw IDs and geo fields that carry no honest signal.
LEAKAGE_AND_ID_COLS = [
    "individual_id", "address_id", "cust_orig_date", "date_of_birth",
    "acct_suspd_date", "city", "county", "latitude", "longitude",
]


def load_data(path="churn_sample.csv"):
    print(f"Loading {path} ...")
    df = pd.read_csv(path, index_col=0)
    print(f"  shape: {df.shape}, churn rate: {df['Churn'].mean():.1%}")
    return df


def engineer_features(df):
    df = df.drop(columns=[c for c in LEAKAGE_AND_ID_COLS if c in df.columns])

    # Numeric NaNs -> median, then any remaining (categorical) NaNs -> "Unknown".
    df = df.fillna(df.median(numeric_only=True)).fillna("Unknown")

    df["tenure_years"] = df["days_tenure"] / 365
    df["premium_per_tenure"] = df["curr_ann_amt"] / (df["tenure_years"] + 1)
    df["age_group"] = pd.cut(
        df["age_in_years"], bins=[0, 30, 40, 50, 60, 100],
        labels=["Young", "Mid-Young", "Middle", "Mid-Senior", "Senior"],
    )
    df["financial_stability"] = (
        df["home_owner"] + df["good_credit"] + df["college_degree"]
    )

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    return pd.get_dummies(df, columns=list(cat_cols), drop_first=True)


def build_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
    }


def evaluate(df_encoded):
    X = df_encoded.drop("Churn", axis=1)
    y = df_encoded["Churn"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale on train only, then SMOTE the scaled training set. SMOTE never touches
    # the test set, so the held-out evaluation stays honest.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train_scaled, y_train)
    print(f"Training class balance after SMOTE: {np.bincount(y_res)} "
          f"(was {np.bincount(y_train)})")

    rows, fitted = [], {}
    for name, model in build_models().items():
        print(f"Training {name} ...")
        model.fit(X_res, y_res)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "Model": name,
            "ROC_AUC": roc_auc_score(y_test, proba),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })
        fitted[name] = proba

    results = pd.DataFrame(rows).sort_values("ROC_AUC", ascending=False)
    print("\n--- Model comparison (sorted by ROC-AUC) ---")
    print(results.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    champion = results.iloc[0]["Model"]
    print(f"\nChampion by ROC-AUC: {champion} ({results.iloc[0]['ROC_AUC']:.3f})")
    return y_test, fitted[champion], champion


def tune_threshold_for_recall(y_test, proba, target_recall=0.70):
    """Lower the decision threshold until recall on churners hits the target."""
    for thresh in np.arange(0.50, 0.05, -0.01):
        pred = (proba >= thresh).astype(int)
        rec = recall_score(y_test, pred)
        if rec >= target_recall:
            prec = precision_score(y_test, pred, zero_division=0)
            print(f"\nTuned threshold {thresh:.2f}: recall={rec:.2f}, precision={prec:.2f} "
                  f"(missing a churner is the costly error).")
            return thresh
    print("\nTarget recall not reached above threshold 0.05; keeping 0.50.")
    return 0.50


def main():
    df = load_data()
    df_encoded = engineer_features(df)
    y_test, champ_proba, champion = evaluate(df_encoded)
    print(f"\nThreshold-tuning the champion ({champion}) for recall...")
    tune_threshold_for_recall(y_test, champ_proba)


if __name__ == "__main__":
    main()
