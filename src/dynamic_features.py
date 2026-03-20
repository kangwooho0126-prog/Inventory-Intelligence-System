import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


def load_encoder_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Encoder file not found: {model_path}")

    encoder = load_model(model_path)
    print(f"Encoder loaded successfully from: {model_path}")
    return encoder


def load_sales_data(sales_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(sales_csv_path):
        raise FileNotFoundError(f"Sales file not found: {sales_csv_path}")

    sales_df = pd.read_csv(sales_csv_path)

    if "item_id" not in sales_df.columns:
        raise ValueError("Sales CSV must contain 'item_id' column.")

    sales_df["item_id"] = sales_df["item_id"].astype(str).str.strip()
    return sales_df


def extract_dynamic_embeddings(
    encoder,
    sales_df: pd.DataFrame,
    item_id_col: str = "item_id"
) -> pd.DataFrame:
    
    numeric_cols = sales_df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("No numeric sales columns found in sales CSV.")

    X_sales = sales_df[numeric_cols].values.astype(float)

   
    X_sales = X_sales.reshape((X_sales.shape[0], X_sales.shape[1], 1))

    X_dynamic = encoder.predict(X_sales, verbose=0)

    dynamic_cols = [f"dyn_{i+1}" for i in range(X_dynamic.shape[1])]
    dynamic_df = pd.DataFrame(X_dynamic, columns=dynamic_cols)
    dynamic_df.insert(0, item_id_col, sales_df[item_id_col].values)

    return dynamic_df