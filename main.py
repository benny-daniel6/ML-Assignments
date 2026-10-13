"""
Main execution script: Runs Boosting, Classification, and Clustering pipelines,
generates comparison tables, displays results, and exports summary files.
"""

import pandas as pd
from boosting_models import train_boosting_models
from classification_models import train_classification_models
from clustering_models import train_clustering_models

def run_all():
    print("=" * 80)
    print(" MACHINE LEARNING MODEL BENCHMARKING: BOOSTING, CLASSIFICATION & CLUSTERING")
    print("=" * 80)
    
    # 1. Boosting
    print("\n[1/3] Running Boosting Models on 'insurance_pre.csv'...")
    df_boosting = train_boosting_models()
    df_boosting.to_csv('results_boosting.csv', index=False)
    print("\n--- Boosting Results ---")
    print(df_boosting.to_string(index=False))
    best_boosting = df_boosting.iloc[0]
    print(f">> Best Boosting Model: {best_boosting['Algorithm']} (R2 Score = {best_boosting['R2 Score']})")
    
    # 2. Classification
    print("\n[2/3] Running Classification Models on 'Social_Network_Ads.csv'...")
    df_classification = train_classification_models()
    df_classification.to_csv('results_classification.csv', index=False)
    print("\n--- Classification Results ---")
    print(df_classification.to_string(index=False))
    best_classification = df_classification.iloc[0]
    print(f">> Best Classification Model: {best_classification['Algorithm']} (Accuracy = {best_classification['Accuracy']}, F1 = {best_classification['F1-Score']})")
    
    # 3. Clustering
    print("\n[3/3] Running Clustering Models on 'Mall_Customers.csv'...")
    df_clustering = train_clustering_models()
    df_clustering.to_csv('results_clustering.csv', index=False)
    print("\n--- Clustering Results ---")
    print(df_clustering.to_string(index=False))
    best_clustering = df_clustering.iloc[0]
    print(f">> Best Clustering Model: {best_clustering['Algorithm']} (Silhouette Score = {best_clustering['Silhouette Score']})")
    
    # 4. Generate Master Markdown Summary
    summary_md = f"""# Machine Learning Model Benchmarking Results

This repository contains implementations and benchmark comparisons for three foundational machine learning paradigms across three distinct datasets:

1. **Boosting (Regression)** on `insurance_pre.csv`
2. **Classification** on `Social_Network_Ads.csv`
3. **Clustering** on `Mall_Customers.csv`

---

## 1. Boosting Results (`insurance_pre.csv`)
- **Task**: Predict medical insurance charges.
- **Evaluation Metric**: $R^2$ Score (higher is better), RMSE, MAE.

{df_boosting.to_markdown(index=False)}

- **Best Algorithm**: **{best_boosting['Algorithm']}** ($R^2$ Score = **{best_boosting['R2 Score']}**, RMSE = **{best_boosting['RMSE']}**)

---

## 2. Classification Results (`Social_Network_Ads.csv`)
- **Task**: Predict whether a user purchased a product based on Age, Gender, and Estimated Salary.
- **Evaluation Metric**: Accuracy (higher is better), F1-Score, Precision, Recall, ROC-AUC.

{df_classification.to_markdown(index=False)}

- **Best Algorithm**: **{best_classification['Algorithm']}** (Accuracy = **{best_classification['Accuracy']}**, F1-Score = **{best_classification['F1-Score']}**)

---

## 3. Clustering Results (`Mall_Customers.csv`)
- **Task**: Segment mall customers based on Annual Income and Spending Score.
- **Evaluation Metric**: Silhouette Score (higher is better), Davies-Bouldin Index (lower is better), Calinski-Harabasz Score.

{df_clustering.to_markdown(index=False)}

- **Best Algorithm**: **{best_clustering['Algorithm']}** (Silhouette Score = **{best_clustering['Silhouette Score']}**, Davies-Bouldin Index = **{best_clustering['Davies-Bouldin Index']}**)

---

## 🏆 Summary of Best Performing Models

| Paradigm | Dataset | Best Performing Algorithm | Key Result Value |
|---|---|---|---|
| **Boosting** | `insurance_pre.csv` | **{best_boosting['Algorithm']}** | $R^2$ Score = **{best_boosting['R2 Score']}** |
| **Classification** | `Social_Network_Ads.csv` | **{best_classification['Algorithm']}** | Accuracy = **{best_classification['Accuracy']}** (F1 = **{best_classification['F1-Score']}**) |
| **Clustering** | `Mall_Customers.csv` | **{best_clustering['Algorithm']}** | Silhouette Score = **{best_clustering['Silhouette Score']}** |
"""
    with open('RESULTS_SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write(summary_md)
        
    print("\n" + "=" * 80)
    print(" Benchmark completed successfully! Results saved to CSV and RESULTS_SUMMARY.md")
    print("=" * 80)

if __name__ == "__main__":
    run_all()
