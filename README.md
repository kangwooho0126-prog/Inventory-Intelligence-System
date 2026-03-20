# slow-moving-inventory-clustering-and-optimization

##  Project Overview
This repository provides an end-to-end pipeline for managing **slow-moving and intermittent demand inventory**, utilizing the **M5 Competition dataset**. 

Managing slow-moving items is a critical challenge in warehouse operations due to high demand uncertainty and holding costs. This project implements a **Divide-and-Conquer strategy** to segment demand patterns and apply specialized forecasting models to optimize inventory policies.

##  Core Pipeline (The "Clustering-to-Strategy" Workflow)
The framework consists of three integrated modules:

1.  **Demand Pattern Clustering**: 
    * Utilizes K-Means and CNN-Transformer autoencoders to isolate **slow-moving (intermittent)** SKUs from high-rotation items.
    * Identifies 7 distinct demand clusters to enable targeted management.

2.  **Specialized Forecasting Engine**:
    * Implements a multi-model approach: **Croston’s Method** for sparse data, **Prophet** for seasonality, and **LightGBM** for non-linear trends.
    * Significantly improves accuracy over traditional baseline models (SMA).

3.  **Inventory Policy Optimization**:
    * Quantifies the financial impact by calculating **Safety Stock (SS)** requirements based on forecast error (RMSE).
    * Demonstrates substantial cost reductions in holding costs for low-rotation inventory.

##  Key Research Impacts
* **Cost Efficiency**: Achieved a quantifiable reduction in safety stock levels while maintaining service targets.
* **Operational Insight**: Proves that SKU-level granularity in clustering leads to more robust warehouse decision-making.

##  Getting Started
1. **Prerequisites**: `pip install pandas numpy lightgbm matplotlib`
2. **Execution**: Run the main pipeline via `python main.py`
3. **Evaluation**: Check `/results` folder for detailed SKU-level metrics and financial impact reports.