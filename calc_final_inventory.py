import pandas as pd
import numpy as np
import os

def calculate_safety_stock(rmse, z_score=1.65, lead_time=7):
    """
    Calculate Safety Stock using RMSE as a proxy for demand uncertainty.
    z_score 1.65 corresponds to a 95% service level.
    """
    return z_score * rmse * np.sqrt(lead_time)

def main():
    # Define data path
    path = "results/forecasting/evaluation_metrics_by_sku.csv"
    if not os.path.exists(path):
        print(f"Error: File not found at {path}. Please check the directory.")
        return

    # Load SKU-level evaluation metrics
    df = pd.read_csv(path)
    
    # Financial assumption: Average unit holding cost ($)
    unit_cost = 15.0  
    
    # Calculate Safety Stock (SS) for both specialized and baseline models
    df['ss_spec'] = calculate_safety_stock(df['rmse_spec'])
    df['ss_base'] = calculate_safety_stock(df['rmse_base'])
    
    # Quantify financial savings from improved forecasting accuracy
    df['saving'] = (df['ss_base'] - df['ss_spec']) * unit_cost
    
    total_money_saved = df['saving'].sum()
    
    # Generate Professional Impact Report
    print("\n" + "="*60)
    print("      PROJECT IMPACT REPORT: INVENTORY OPTIMIZATION")
    print("="*60)
    print(f"Total SKUs Evaluated          : {len(df)}")
    print(f"Total Inventory Cost Savings  : ${total_money_saved:,.2f}")
    print("="*60)
    print("Optimization Insight: Accuracy improvements directly reduce")
    print("safety stock requirements and associated holding costs.")
    print("="*60)

if __name__ == "__main__":
    main()