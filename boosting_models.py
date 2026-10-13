"""
Boosting Models on insurance_pre.csv
Predicting medical insurance charges using boosting regression algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def train_boosting_models():
    # 1. Load Data
    df = pd.read_csv('insurance_pre.csv')
    
    # 2. Simple Preprocessing
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # 3. Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. Define Boosting Models
    models = {
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "AdaBoost": AdaBoostRegressor(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42, verbosity=0),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0)
    }
    
    # 5. Train & Evaluate
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        results.append({
            "Algorithm": name,
            "R2 Score": round(r2, 4),
            "RMSE": round(rmse, 2),
            "MAE": round(mae, 2)
        })
        
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    return results_df

if __name__ == "__main__":
    df_results = train_boosting_models()
    print("=== BOOSTING ALGORITHM RESULTS (insurance_pre.csv) ===")
    print(df_results.to_string(index=False))
    best_model = df_results.iloc[0]
    print(f"\nBest Model: {best_model['Algorithm']} with R2 Score = {best_model['R2 Score']}")
