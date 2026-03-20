import sys
import os
import pandas as pd


project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.feature_fusion import (
    load_static_features,
    load_dynamic_features,
    fuse_static_dynamic_features
)

from src.clustering import run_kmeans_multiple_k

from src.visualization import (
    load_sales_data,
    load_cluster_assignments,
    plot_cluster_mean_patterns
)

from src.pattern_analysis import (
    compute_cluster_statistics,
    assign_pattern_labels
)

from src.inventory_decision import inventory_decision


def main():
    static_csv_path = "results/static_features_12d.csv"
    dynamic_csv_path = "data/dynamic_features_16d.csv"
    sales_csv_path = "data/m5_sales_subset.csv"

   
    static_df = load_static_features(static_csv_path)
    dynamic_df = load_dynamic_features(dynamic_csv_path)

   
    fused_df = fuse_static_dynamic_features(static_df, dynamic_df)

   
    metrics_df, best_results = run_kmeans_multiple_k(
        fused_df,
        k_range=range(3, 9),
        n_runs=20
    )

    
    metrics_output_path = "results/fused_clustering_metrics.csv"
    metrics_df.to_csv(metrics_output_path, index=False)
    print(f"Saved: {metrics_output_path}")

    
    for k, result in best_results.items():
        save_path = f"results/cluster_assignments_k{k}.csv"
        result["assignment_df"].to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    
    selected_k = 7
    assign_csv_path = f"results/cluster_assignments_k{selected_k}.csv"

    sales_df = load_sales_data(sales_csv_path)
    assign_df = load_cluster_assignments(assign_csv_path)

    plot_cluster_mean_patterns(
        sales_df,
        assign_df,
        save_path=f"results/cluster_patterns_k{selected_k}.png"
    )

   
    stats_df = compute_cluster_statistics(sales_df, assign_df)
    stats_df = assign_pattern_labels(stats_df)

    print("stats_df columns:", list(stats_df.columns))

    pattern_summary_path = "results/cluster_pattern_summary.csv"
    stats_df.to_csv(pattern_summary_path, index=False)
    print(f"Saved: {pattern_summary_path}")

   
    if "pattern_label" in stats_df.columns:
        pattern_col = "pattern_label"
    elif "pattern" in stats_df.columns:
        pattern_col = "pattern"
    elif "pattern_type" in stats_df.columns:
        pattern_col = "pattern_type"
    else:
        raise KeyError(
            f"No pattern column found in stats_df. Available columns: {list(stats_df.columns)}"
        )

    merged_df = assign_df.merge(
        stats_df[["cluster", pattern_col]],
        on="cluster",
        how="left"
    )

    inventory_results = []

    for _, row in merged_df.iterrows():
        pattern = row[pattern_col]

        
        mean_demand = 50
        std = 10
        lead_time = 7

        decision = inventory_decision(
            mean_demand=mean_demand,
            std=std,
            lead_time=lead_time,
            pattern=pattern
        )

        inventory_results.append({
            "item_id": row["item_id"],
            "cluster": row["cluster"],
            "pattern": pattern,
            "mean_demand": mean_demand,
            "std": std,
            "lead_time": lead_time,
            "safety_stock": decision["safety_stock"],
            "reorder_point": decision["reorder_point"]
        })

    inventory_df = pd.DataFrame(inventory_results)
    inventory_output_path = f"results/inventory_decision_k{selected_k}.csv"
    inventory_df.to_csv(inventory_output_path, index=False)
    print(f"Saved: {inventory_output_path}")


if __name__ == "__main__":
    main()