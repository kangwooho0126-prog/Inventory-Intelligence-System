import os
import pandas as pd


def load_static_features(static_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(static_csv_path):
        raise FileNotFoundError(f"Static feature file not found: {static_csv_path}")

    static_df = pd.read_csv(static_csv_path)
    static_df["item_id"] = static_df["item_id"].astype(str).str.strip()
    return static_df


def load_dynamic_features(dynamic_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(dynamic_csv_path):
        raise FileNotFoundError(f"Dynamic feature file not found: {dynamic_csv_path}")

    dynamic_df = pd.read_csv(dynamic_csv_path)
    dynamic_df["item_id"] = dynamic_df["item_id"].astype(str).str.strip()
    return dynamic_df


def fuse_static_dynamic_features(
    static_df: pd.DataFrame,
    dynamic_df: pd.DataFrame
) -> pd.DataFrame:
    fused_df = static_df.merge(dynamic_df, on="item_id", how="inner")
    print(f"Matched SKUs after fusion: {len(fused_df)}")
    return fused_df