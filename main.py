import sys
import os

# 核心修复：获取 main.py 所在的绝对目录，并强行插入到 Python 搜索路径的最前面
# 这样可以确保 Python 绝对优先从当前项目的 src 文件夹找模块，避免任何路径遗失或命名冲突
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

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


def main():
    static_csv_path = "results/static_features_12d.csv"
    dynamic_csv_path = "data/dynamic_features_16d.csv"
    sales_csv_path = "data/m5_sales_subset.csv"

    # 1. load features
    static_df = load_static_features(static_csv_path)
    dynamic_df = load_dynamic_features(dynamic_csv_path)

    # 2. fuse
    fused_df = fuse_static_dynamic_features(static_df, dynamic_df)

    # 3. clustering
    metrics_df, best_results = run_kmeans_multiple_k(
        fused_df,
        k_range=range(3, 9),
        n_runs=20
    )

    # 4. save metrics
    metrics_df.to_csv("results/fused_clustering_metrics.csv", index=False)
    print("\nSaved: results/fused_clustering_metrics.csv")
    print(metrics_df)

    # 5. save assignments
    for k, result in best_results.items():
        save_path = f"results/cluster_assignments_k{k}.csv"
        result["assignment_df"].to_csv(save_path, index=False)
        print(f"Saved: {save_path}")

    # 6. visualization (use K=7)
    assign_csv_path = "results/cluster_assignments_k7.csv"

    sales_df = load_sales_data(sales_csv_path)
    assign_df = load_cluster_assignments(assign_csv_path)

    plot_cluster_mean_patterns(
        sales_df,
        assign_df,
        save_path="results/cluster_patterns_k7.png"
    )

    # 7. pattern analysis
    print("\n=== Pattern Analysis ===")

    stats_df = compute_cluster_statistics(sales_df, assign_df)
    stats_df = assign_pattern_labels(stats_df)

    stats_df.to_csv("results/cluster_pattern_summary.csv", index=False)

    print("\nCluster Pattern Summary:")
    print(stats_df)


if __name__ == "__main__":
    main()