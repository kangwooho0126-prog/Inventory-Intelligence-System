import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import warnings

warnings.filterwarnings('ignore')

def calc_wape(y_true, y_pred):
    sum_actual = np.sum(np.abs(y_true))
    if sum_actual == 0: return 0.0 if np.sum(np.abs(y_pred)) == 0 else 1.0
    return np.sum(np.abs(y_true - y_pred)) / sum_actual

def fix_columns(df):
    """自动将包含特定关键字的列名统一化"""
    mapping = {}
    for col in df.columns:
        c_low = col.lower()
        if 'cluster' in c_low or 'label' in c_low: mapping[col] = 'cluster'
        elif 'method' in c_low: mapping[col] = 'method'
        elif 'wape' in c_low and 'base' in c_low: mapping[col] = 'wape_base'
        elif 'wape' in c_low and 'spec' in c_low: mapping[col] = 'wape_spec'
        elif 'wape' in c_low and 'lgbm' in c_low: mapping[col] = 'wape_lgbm'
    return df.rename(columns=mapping)

def main():
    # 1. 路径检查
    old_summary_path = "results/forecasting/evaluation_summary_by_cluster.csv"
    sales_path = "data/m5_sales_subset.csv"
    assign_path = "results/cluster_assignments_k7.csv"

    # 2. 加载与列名标准化
    sales_df = pd.read_csv(sales_path)
    assign_df = fix_columns(pd.read_csv(assign_path))
    old_summary = fix_columns(pd.read_csv(old_summary_path))
    
    day_cols = [c for c in sales_df.columns if c.startswith('Day_')]
    h = 28
    
    # 3. 构造特征
    print("正在构造 LightGBM 特征...")
    data_list = []
    for _, row in sales_df.iterrows():
        item_id = row['item_id']
        ts = row[day_cols].values
        c_match = assign_df[assign_df['item_id'] == item_id]
        c_val = c_match['cluster'].values[0] if not c_match.empty else 0
        
        data_list.append({
            'item_id': item_id,
            'lag_28': ts[-(h+1)],
            'rolling_mean_7': ts[-(h+7):-h].mean(),
            'cluster': c_val,
            'target': ts[-h:].mean()
        })
    
    df_feat = pd.DataFrame(data_list)
    X = df_feat[['lag_28', 'rolling_mean_7', 'cluster']].copy()
    X['cluster'] = X['cluster'].astype('category')
    y = df_feat['target']
    
    # 4. 训练与预测
    model = lgb.train({'objective':'regression','verbosity':-1}, lgb.Dataset(X, label=y))
    df_feat['y_pred_lgbm'] = model.predict(X)

    # 5. 计算 WAPE
    lgbm_res = []
    for _, row in df_feat.iterrows():
        actual = sales_df[sales_df['item_id'] == row['item_id']][day_cols].values[0][-h:]
        lgbm_res.append({'cluster': row['cluster'], 'wape_lgbm': calc_wape(actual, np.full(h, row['y_pred_lgbm']))})
    
    new_summary = pd.DataFrame(lgbm_res).groupby('cluster')['wape_lgbm'].mean().reset_index()

    # 6. 合并 (使用 how='inner' 确保只保留对齐的行)
    final_comparison = pd.merge(old_summary, new_summary, on='cluster', how='inner')

    # 7. 打印与保存
    print("\n" + "="*65)
    print("             三方模型对比最终报告 (WAPE)")
    print("="*65)
    # 只打印存在的列，防止报错
    cols_to_show = [c for c in ['cluster', 'method', 'wape_spec', 'wape_base', 'wape_lgbm'] if c in final_comparison.columns]
    print(final_comparison[cols_to_show].to_string(index=False))
    print("="*65)
    
    save_path = "results/forecasting/final_comparison_with_lgbm.csv"
    final_comparison.to_csv(save_path, index=False)
    print(f"✅ 保存成功！路径: {save_path}")

if __name__ == "__main__":
    main()