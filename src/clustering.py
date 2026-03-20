import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans_multiple_k(
    fused_df: pd.DataFrame,
    k_range=range(3, 9),
    n_runs: int = 20,
    item_id_col: str = "item_id"
):
    feature_cols = [c for c in fused_df.columns if c != item_id_col]
    X = fused_df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics_rows = []
    best_results = {}

    for k in k_range:
        best_silhouette = -1
        best_labels = None
        best_model = None

        for seed in range(n_runs):
            kmeans = KMeans(
                n_clusters=k,
                random_state=seed,
                n_init=10
            )
            labels = kmeans.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, labels)

            if sil > best_silhouette:
                best_silhouette = sil
                best_labels = labels
                best_model = kmeans

        metrics_rows.append({
            "k": k,
            "best_silhouette": best_silhouette
        })

        assignment_df = fused_df[[item_id_col]].copy()
        assignment_df["cluster"] = best_labels

        best_results[k] = {
            "assignment_df": assignment_df,
            "model": best_model
        }

    metrics_df = pd.DataFrame(metrics_rows)
    return metrics_df, best_results