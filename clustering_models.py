"""
Clustering Models on Mall_Customers.csv
Segmenting mall customers using unsupervised clustering algorithms based on Annual Income and Spending Score.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    DBSCAN,
    MeanShift,
    Birch,
    AffinityPropagation
)
from sklearn.mixture import GaussianMixture

def train_clustering_models():
    # 1. Load Data
    df = pd.read_csv('Mall_Customers.csv')
    
    # 2. Select Features (Standard benchmark: Annual Income and Spending Score)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    # 3. Scale Features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Define Clustering Models
    # Note: 5 clusters is the well-known optimal ground truth for Mall Customers (Elbow/Silhouette)
    models = {
        "K-Means (k=5)": KMeans(n_clusters=5, random_state=42, n_init=10),
        "Agglomerative Clustering (k=5)": AgglomerativeClustering(n_clusters=5),
        "Gaussian Mixture (k=5)": GaussianMixture(n_components=5, random_state=42),
        "Birch (k=5)": Birch(n_clusters=5),
        "Mean Shift": MeanShift(),
        "Affinity Propagation": AffinityPropagation(random_state=42),
        "DBSCAN": DBSCAN(eps=0.35, min_samples=5)
    }
    
    # 5. Fit & Evaluate
    results = []
    for name, model in models.items():
        if isinstance(model, GaussianMixture):
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)
            
        # Ignore noise points (-1) in DBSCAN for fair metric computation if present
        valid_mask = labels != -1
        unique_labels = len(set(labels[valid_mask]))
        
        # Silhouette requires at least 2 distinct clusters
        if unique_labels > 1:
            sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
            db = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
            ch = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
        else:
            sil, db, ch = -1.0, float('nan'), float('nan')
            
        results.append({
            "Algorithm": name,
            "Clusters Found": unique_labels,
            "Silhouette Score": round(sil, 4),
            "Davies-Bouldin Index": round(db, 4) if not np.isnan(db) else "N/A",
            "Calinski-Harabasz Score": round(ch, 2) if not np.isnan(ch) else "N/A"
        })
        
    results_df = pd.DataFrame(results).sort_values(by="Silhouette Score", ascending=False).reset_index(drop=True)
    return results_df

if __name__ == "__main__":
    df_results = train_clustering_models()
    print("=== CLUSTERING ALGORITHM RESULTS (Mall_Customers.csv) ===")
    print(df_results.to_string(index=False))
    best_model = df_results.iloc[0]
    print(f"\nBest Model: {best_model['Algorithm']} with Silhouette Score = {best_model['Silhouette Score']}")
