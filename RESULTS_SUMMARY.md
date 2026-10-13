# Machine Learning Model Benchmarking Results

This repository contains implementations and benchmark comparisons for three foundational machine learning paradigms across three distinct datasets:

1. **Boosting (Regression)** on `insurance_pre.csv`
2. **Classification** on `Social_Network_Ads.csv`
3. **Clustering** on `Mall_Customers.csv`

---

## 1. Boosting Results (`insurance_pre.csv`)
- **Task**: Predict medical insurance charges.
- **Evaluation Metric**: $R^2$ Score (higher is better), RMSE, MAE.

| Algorithm            |   R2 Score |    RMSE |     MAE |
|:---------------------|-----------:|--------:|--------:|
| Gradient Boosting    |     0.8785 | 4342.91 | 2424.53 |
| LightGBM             |     0.8676 | 4533.41 | 2559.43 |
| CatBoost             |     0.8662 | 4557.49 | 2534.3  |
| HistGradientBoosting |     0.8632 | 4607.72 | 2577.51 |
| XGBoost              |     0.8434 | 4931.26 | 2783.73 |
| AdaBoost             |     0.827  | 5181.86 | 4253.85 |

- **Best Algorithm**: **Gradient Boosting** ($R^2$ Score = **0.8785**, RMSE = **4342.91**)

---

## 2. Classification Results (`Social_Network_Ads.csv`)
- **Task**: Predict whether a user purchased a product based on Age, Gender, and Estimated Salary.
- **Evaluation Metric**: Accuracy (higher is better), F1-Score, Precision, Recall, ROC-AUC.

| Algorithm                    |   Accuracy |   F1-Score |   Precision |   Recall |   ROC-AUC |
|:-----------------------------|-----------:|-----------:|------------:|---------:|----------:|
| Naive Bayes                  |     0.9375 |     0.9091 |      0.9259 |   0.8929 |    0.9856 |
| K-Nearest Neighbors          |     0.925  |     0.8966 |      0.8667 |   0.9286 |    0.9533 |
| Support Vector Machine (RBF) |     0.925  |     0.8966 |      0.8667 |   0.9286 |    0.9677 |
| Logistic Regression          |     0.8875 |     0.8235 |      0.913  |   0.75   |    0.9705 |
| Random Forest                |     0.875  |     0.8214 |      0.8214 |   0.8214 |    0.9454 |
| Gradient Boosting            |     0.875  |     0.8148 |      0.8462 |   0.7857 |    0.9478 |
| XGBoost                      |     0.875  |     0.8214 |      0.8214 |   0.8214 |    0.9478 |
| Decision Tree                |     0.825  |     0.75   |      0.75   |   0.75   |    0.8077 |

- **Best Algorithm**: **Naive Bayes** (Accuracy = **0.9375**, F1-Score = **0.9091**)

---

## 3. Clustering Results (`Mall_Customers.csv`)
- **Task**: Segment mall customers based on Annual Income and Spending Score.
- **Evaluation Metric**: Silhouette Score (higher is better), Davies-Bouldin Index (lower is better), Calinski-Harabasz Score.

| Algorithm                      |   Clusters Found |   Silhouette Score | Davies-Bouldin Index   | Calinski-Harabasz Score   |
|:-------------------------------|-----------------:|-------------------:|:-----------------------|:--------------------------|
| DBSCAN                         |                6 |             0.5577 | 0.5106                 | 245.36                    |
| K-Means (k=5)                  |                5 |             0.5547 | 0.5722                 | 248.65                    |
| Agglomerative Clustering (k=5) |                5 |             0.5538 | 0.5779                 | 244.41                    |
| Gaussian Mixture (k=5)         |                5 |             0.5537 | 0.576                  | 244.94                    |
| Affinity Propagation           |               11 |             0.4321 | 0.7513                 | 265.08                    |
| Birch (k=5)                    |                5 |             0.4145 | 0.6979                 | 125.95                    |
| Mean Shift                     |                1 |            -1      | N/A                    | N/A                       |

- **Best Algorithm**: **DBSCAN** (Silhouette Score = **0.5577**, Davies-Bouldin Index = **0.5106**)

---

## 🏆 Summary of Best Performing Models

| Paradigm | Dataset | Best Performing Algorithm | Key Result Value |
|---|---|---|---|
| **Boosting** | `insurance_pre.csv` | **Gradient Boosting** | $R^2$ Score = **0.8785** |
| **Classification** | `Social_Network_Ads.csv` | **Naive Bayes** | Accuracy = **0.9375** (F1 = **0.9091**) |
| **Clustering** | `Mall_Customers.csv` | **DBSCAN** | Silhouette Score = **0.5577** |
