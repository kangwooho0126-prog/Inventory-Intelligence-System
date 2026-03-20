import pandas as pd
import numpy as np
import os

def calculate_safety_stock(rmse, z_score=1.65, lead_time=7):
    # 安全库存公式: Z * RMSE * sqrt(L/T)
    return z_score * rmse * np.sqrt(lead_time)

def main():
    # 读取你跑出来的明细表
    path = "results/forecasting/evaluation_metrics_by_sku.csv"
    if not os.path.exists(path):
        print("❌ 找不到明细表，请检查路径。")
        return

    df = pd.read_csv(path)
    
    # 业务假设
    unit_cost = 15.0  # 假设每个SKU平均价值 15元
    
    # 计算两种策略的安全库存量
    df['ss_spec'] = calculate_safety_stock(df['rmse_spec'])
    df['ss_base'] = calculate_safety_stock(df['rmse_base'])
    
    # 计算节省的钱
    df['saving'] = (df['ss_base'] - df['ss_spec']) * unit_cost
    
    total_money_saved = df['saving'].sum()
    
    print("\n" + "="*50)
    print("🚀 最终项目成果汇报")
    print("="*50)
    print(f"参与评估的 SKU 总数: {len(df)}")
    print(f"预测精度提升带来的库存节省总额: ${total_money_saved:,.2f}")
    print("="*50)
    print("💡 建议将此数字写进简历的开头第一句话！")

if __name__ == "__main__":
    main()
    