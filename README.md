# Machine Learning Benchmarking: Boosting, Classification & Clustering

This repository contains clean, modular, and easy-to-run Python implementations comparing multiple algorithms across three machine learning paradigms on 3 standard datasets:

1. **Boosting (Regression)** on `insurance_pre.csv`
2. **Classification** on `Social_Network_Ads.csv`
3. **Clustering** on `Mall_Customers.csv`

---

## 📊 Summary of Best Performing Models

| Paradigm | Dataset | Best Performing Algorithm | Best Score / Metric |
|---|---|---|---|
| **Boosting** | `insurance_pre.csv` | **Gradient Boosting Regressor** | **$R^2$ = 0.8785** (RMSE = 4,342.91, MAE = 2,424.53) |
| **Classification** | `Social_Network_Ads.csv` | **Naive Bayes (GaussianNB)** | **Accuracy = 93.75%** (F1 = 0.9091, ROC-AUC = 0.9856) |
| **Clustering** | `Mall_Customers.csv` | **DBSCAN** | **Silhouette Score = 0.5577** (Davies-Bouldin = 0.5106) |

---

## 📈 Detailed Results

### 1. Boosting Algorithms (`insurance_pre.csv`)
- **Objective**: Predict insurance charges based on age, sex, bmi, children, and smoker status.
- **Metric**: $R^2$ Score (higher is better), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).

| Rank | Algorithm | $R^2$ Score | RMSE | MAE |
|:---:|---|:---:|:---:|:---:|
| 🥇 | **Gradient Boosting** | **0.8785** | **4342.91** | **2424.53** |
| 🥈 | LightGBM | 0.8676 | 4533.41 | 2559.43 |
| 🥉 | CatBoost | 0.8662 | 4557.49 | 2534.30 |
| 4 | HistGradientBoosting | 0.8632 | 4607.72 | 2577.51 |
| 5 | XGBoost | 0.8434 | 4931.26 | 2783.73 |
| 6 | AdaBoost | 0.8270 | 5181.86 | 4253.85 |

> **Winner**: **Gradient Boosting Regressor** achieved the highest $R^2$ score of **0.8785** and lowest RMSE of **4,342.91**.

---

### 2. Classification Algorithms (`Social_Network_Ads.csv`)
- **Objective**: Predict whether a user purchased an item (0 = No, 1 = Yes) using Age, Gender, and Estimated Salary.
- **Metric**: Accuracy, F1-Score, Precision, Recall, ROC-AUC.

| Rank | Algorithm | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Naive Bayes** | **0.9375** | **0.9091** | 0.9259 | 0.8929 | **0.9856** |
| 🥈 | K-Nearest Neighbors | 0.9250 | 0.8966 | 0.8667 | 0.9286 | 0.9533 |
| 🥉 | Support Vector Machine (RBF) | 0.9250 | 0.8966 | 0.8667 | 0.9286 | 0.9677 |
| 4 | Logistic Regression | 0.8875 | 0.8235 | 0.9130 | 0.7500 | 0.9705 |
| 5 | Random Forest | 0.8750 | 0.8214 | 0.8214 | 0.8214 | 0.9454 |
| 6 | Gradient Boosting | 0.8750 | 0.8148 | 0.8462 | 0.7857 | 0.9478 |
| 7 | XGBoost | 0.8750 | 0.8214 | 0.8214 | 0.8214 | 0.9478 |
| 8 | Decision Tree | 0.8250 | 0.7500 | 0.7500 | 0.7500 | 0.8077 |

> **Winner**: **Naive Bayes** achieved the highest Accuracy of **93.75%**, F1-Score of **0.9091**, and ROC-AUC of **0.9856**.

---

### 3. Clustering Algorithms (`Mall_Customers.csv`)
- **Objective**: Identify distinct customer clusters based on Annual Income and Spending Score.
- **Metric**: Silhouette Score (higher is better), Davies-Bouldin Index (lower is better), Calinski-Harabasz Score.

| Rank | Algorithm | Clusters Found | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Score |
|:---:|---|:---:|:---:|:---:|:---:|
| 🥇 | **DBSCAN** | 6 | **0.5577** | **0.5106** | 245.36 |
| 🥈 | K-Means (k=5) | 5 | 0.5547 | 0.5722 | **248.65** |
| 🥉 | Agglomerative Clustering (k=5) | 5 | 0.5538 | 0.5779 | 244.41 |
| 4 | Gaussian Mixture (k=5) | 5 | 0.5537 | 0.5760 | 244.94 |
| 5 | Affinity Propagation | 11 | 0.4321 | 0.7513 | 265.08 |
| 6 | Birch (k=5) | 5 | 0.4145 | 0.6979 | 125.95 |
| 7 | Mean Shift | 1 | -1.0000 | N/A | N/A |

> **Winner**: **DBSCAN** achieved the highest Silhouette Score of **0.5577** and the best (lowest) Davies-Bouldin Index of **0.5106**, closely followed by **K-Means (k=5)** at **0.5547**.

---

## 📁 Repository Structure

```text
├── insurance_pre.csv          # Boosting dataset (Insurance charges)
├── Social_Network_Ads.csv     # Classification dataset (Purchased ads)
├── Mall_Customers.csv         # Clustering dataset (Customer segmentation)
├── boosting_models.py         # Boosting training & evaluation script
├── classification_models.py   # Classification training & evaluation script
├── clustering_models.py       # Clustering training & evaluation script
├── main.py                    # Master script to execute all pipelines & generate summaries
├── results_boosting.csv       # Tabular export for boosting results
├── results_classification.csv # Tabular export for classification results
├── results_clustering.csv     # Tabular export for clustering results
├── RESULTS_SUMMARY.md         # Markdown benchmark summary
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation & results
```

---

## 🚀 How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run all benchmarks at once**:
   ```bash
   python main.py
   ```

3. **Or run individual modules**:
   ```bash
   python boosting_models.py
   python classification_models.py
   python clustering_models.py
   ```
