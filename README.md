# Inventory-Intelligence-System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Forecasting](https://img.shields.io/badge/Forecasting-Demand%20Prediction-orange)
![Clustering](https://img.shields.io/badge/Clustering-Demand%20Patterns-yellow)
![Inventory Optimization](https://img.shields.io/badge/Inventory-Intelligence-green)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-red)
![Research Project](https://img.shields.io/badge/Project-Academic%20Research-purple)

##  Project Overview

This project develops an end-to-end intelligent system for detecting slow-moving and censored demand patterns and optimizing inventory policies.

It integrates:

- Feature Engineering (Statistical + Temporal Representation)
- Unsupervised Clustering (K-Means, multi-K analysis)
- Demand Pattern Recognition
- Forecasting (Baseline + Pattern-aware)
- Inventory Policy Optimization (Pattern-aware)

The goal is to transform raw sales data into actionable inventory decisions.

---

##  Business Problem

In supply chain management, slow-moving and intermittent demand items are difficult to manage due to:

- High demand uncertainty
- Long zero-demand periods
- Demand bursts
- Overstock and stockout risks

Traditional methods treat all SKUs similarly, leading to inefficient inventory policies.

 This project addresses:

> **How to identify demand patterns and optimize inventory decisions accordingly**

---

##  Solution Overview

The system follows a structured pipeline:

Sales Data
   ↓
Feature Extraction (Statistical + Temporal)
   ↓
Feature Fusion
   ↓
Clustering (Multi-K KMeans)
   ↓
Pattern Identification
   ↓
Forecasting (Baseline + Pattern-aware)
   ↓
Inventory Policy Optimization

---

##  Methodology

### 1. Feature Engineering
- Statistical features (mean, std, CV, zero-ratio)
- Temporal features (learned representation)
- Feature fusion (hybrid representation)

---

### 2. Clustering
- KMeans clustering (K=3~8)
- Multiple runs for stability
- Evaluation metrics:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index

---

### 3. Pattern Recognition

Each cluster is interpreted as a demand pattern:

| Pattern       | Description |
|--------------|------------|
| Smooth       | Stable demand |
| Intermittent | Many zero-demand periods |
| Burst        | Sudden spikes |
| Volatile     | High variability |

---

### 4. Forecasting

Two approaches are compared:

- Baseline (moving average)
- Pattern-aware forecasting

Evaluation metrics:
- WAPE
- RMSE

---

### 5. Inventory Policy Optimization (Core Contribution)

A pattern-aware optimization module is introduced:

- Dynamic service level selection
- Safety stock optimization
- Reorder point calculation
- Order-up-to level optimization
- EOQ-based order quantity estimation

Different demand patterns use different cost structures:

| Pattern       | Strategy |
|--------------|--------|
| Intermittent | Conservative inventory |
| Burst        | Higher safety stock |
| Smooth       | Stable replenishment |
| Volatile     | High resilience buffer |

---

##  Project Structure

```text
M5-inventory-intelligence-system/
├── data/
│   ├── raw/
│   └── processed/
│       └── static_features_12d.csv
├── results/
│   ├── clustering/        
│   ├── forecasting/       
│   └── decision/          
├── src/
│   ├── data/            
│   ├── features/         
│   ├── clustering/       
│   ├── forecasting/       
│   ├── decision/         
│   └── optimization/      
├── main.py
├── requirements.txt
└── README.md
```

---

##  How to Run

### 1. Install dependencies

pip install -r requirements.txt

---

### 2. Run full pipeline

python main.py

---

### 3. Optional modules

python src/forecasting/run_forecasting.py
python src/forecasting/train_lightgbm.py
python src/decision/calc_final_inventory.py

---

##  Key Outputs

### Clustering
- cluster_assignments_k*.csv
- cluster_pattern_summary_k7.csv
- cluster_patterns_k7.png

### Forecasting
- evaluation_metrics_by_sku.csv
- evaluation_summary_by_cluster.csv
- final_comparison_with_lgbm.csv

### Inventory Decision & Optimization
- inventory_decision_k7.csv

Includes:
- safety stock (rule-based)
- optimized safety stock
- reorder point
- optimal order quantity
- estimated total cost

---

##  Key Results

- Demand patterns can be effectively segmented using hybrid features
- Pattern-aware forecasting improves performance over baseline
- Inventory policies vary significantly across patterns
- Optimization reduces unnecessary inventory while maintaining service levels

---

##  Key Contributions

1. Hybrid feature representation combining statistical and temporal features  
2. Multi-resolution clustering framework for demand pattern identification  
3. Integration of clustering with forecasting performance evaluation  
4. Pattern-aware inventory policy optimization (core contribution)  
5. End-to-end pipeline from data to decision  

---

##  Skills Demonstrated

- Time Series Analysis
- Clustering (KMeans)
- Feature Engineering
- Demand Forecasting
- Inventory Optimization
- Supply Chain Analytics
- Python (Pandas, NumPy, Scikit-learn, LightGBM)

---

##  Related Project

This project focuses on SKU-level inventory intelligence.

Complementary project:
Multi-Warehouse Replenishment Optimization (network-level optimization)

Together, they form a complete supply chain decision framework:
- Demand pattern recognition
- SKU policy optimization
- Network-level replenishment optimization

---

##  Author

Andrea Kang  
Supply Chain & AI Researcher
