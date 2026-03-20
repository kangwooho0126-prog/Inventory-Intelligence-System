import os
import pandas as pd
from static_features import extract_static_features_for_series


def load_sales_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "item_id" not in df.columns:
        raise ValueError("CSV must contain 'item_id' column.")

    return df


def build_static_feature_dataset(csv_path: str) -> pd.DataFrame:
    sales_df = load_sales_csv(csv_path)

    
    numeric_cols = sales_df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("No numeric sales columns found in the CSV file.")

    results = []

    for _, row in sales_df.iterrows():
        item_id = row["item_id"]
        sales_series = row[numeric_cols].values.astype(float)

        features = extract_static_features_for_series(
            sales_series,
            item_id=item_id
        )

        results.append(features)

    feature_df = pd.DataFrame(results)
    return feature_df


def save_feature_dataset(feature_df: pd.DataFrame, output_path: str):
    feature_df.to_csv(output_path, index=False)
    print(f"Feature dataset saved to: {output_path}")