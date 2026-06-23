import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')

def load_and_sample_data():
    """
    Load data from the provided churn_sample.csv file.
    """
    print("Loading dataset from churn_sample.csv...")
    df = pd.read_csv('churn_sample.csv', index_col=0)
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_and_engineer(df):
    """
    Cleans data and adds engineered features.
    """
    print("Cleaning and engineering features...")
    
    # Drop IDs, geographic, and data leakage columns
    cols_to_drop = ['individual_id', 'address_id', 'cust_orig_date', 'date_of_birth', 
                    'acct_suspd_date', 'city', 'county', 'latitude', 'longitude']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna('Unknown')
    
    # Feature Engineering
    df['tenure_years'] = df['days_tenure'] / 365
    df['premium_per_tenure'] = df['curr_ann_amt'] / (df['tenure_years'] + 1)
    
    # Age Groups
    df['age_group'] = pd.cut(df['age_in_years'], bins=[0, 30, 40, 50, 60, 100], 
                             labels=['Young', 'Mid-Young', 'Middle', 'Mid-Senior', 'Senior'])
    
    # Financial Stability Score
    df['financial_stability'] = df['home_owner'] + df['good_credit'] + df['college_degree']
    
    # Encoding Categoricals
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)
    
    return df_encoded

def train_and_evaluate(df_encoded):
    print("Preparing for modeling...")
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC_AUC': roc
        })
        
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))
    
    best_model = results_df.loc[results_df['ROC_AUC'].idxmax()]
    print(f"\nBest Model by ROC-AUC: {best_model['Model']} ({best_model['ROC_AUC']:.4f})")

if __name__ == "__main__":
    df = load_and_sample_data()
    df_processed = preprocess_and_engineer(df)
    train_and_evaluate(df_processed)
