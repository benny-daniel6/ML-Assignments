"""
Classification Models on Social_Network_Ads.csv
Predicting whether a user purchased a product based on Age, EstimatedSalary, and Gender.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def train_classification_models():
    # 1. Load Data
    df = pd.read_csv('Social_Network_Ads.csv')
    
    # 2. Simple Preprocessing
    # Drop User ID (unique identifier)
    df = df.drop('User ID', axis=1, errors='ignore')
    # Encode categorical Gender
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    
    X = df.drop('Purchased', axis=1)
    y = df['Purchased']
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Feature Scaling (essential for distance/linear models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Define Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine (RBF)": SVC(probability=True, random_state=42),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    # 6. Train & Evaluate
    results = []
    for name, model in models.items():
        # Scale for all for consistent comparison
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        
        results.append({
            "Algorithm": name,
            "Accuracy": round(acc, 4),
            "F1-Score": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "ROC-AUC": round(auc, 4)
        })
        
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    return results_df

if __name__ == "__main__":
    df_results = train_classification_models()
    print("=== CLASSIFICATION ALGORITHM RESULTS (Social_Network_Ads.csv) ===")
    print(df_results.to_string(index=False))
    best_model = df_results.iloc[0]
    print(f"\nBest Model: {best_model['Algorithm']} with Accuracy = {best_model['Accuracy']} (F1 = {best_model['F1-Score']})")
