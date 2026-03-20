import pandas as pd
import numpy as np


def get_day_columns(df):
    return sorted(
        [c for c in df.columns if c.startswith("Day_")],
        key=lambda x: int(x.split("_")[1])
    )


def compute_cluster_statistics(sales_df, assign_df):
    merged = assign_df.merge(sales_df, on="item_id", how="inner")
    day_cols = get_day_columns(sales_df)

    results = []

    for cluster_id in sorted(merged["cluster"].unique()):
        cluster_data = merged[merged["cluster"] == cluster_id]
        X = cluster_data[day_cols].values

        mean_sales = np.mean(X)
        std_sales = np.std(X)
        zero_ratio = np.mean(X == 0)

        cv = std_sales / (mean_sales + 1e-6)

      
        burst_ratio = np.mean(X > 2 * mean_sales)

        results.append({
            "cluster": cluster_id,
            "mean_sales": round(mean_sales, 4),
            "std_sales": round(std_sales, 4),
            "cv": round(cv, 4),
            "zero_ratio": round(zero_ratio, 4),
            "burst_ratio": round(burst_ratio, 4)
        })

    return pd.DataFrame(results)



def classify_pattern(row):
    if row["zero_ratio"] > 0.6:
        return "Intermittent"

    elif row["burst_ratio"] > 0.1:
        return "Burst"

    elif row["cv"] < 0.5:
        return "Smooth"

    else:
        return "Volatile"


def assign_pattern_labels(stats_df):
    stats_df["pattern_type"] = stats_df.apply(classify_pattern, axis=1)
    return stats_df