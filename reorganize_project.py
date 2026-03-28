from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent


TARGET_DIRS = [
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "results" / "clustering",
    ROOT / "results" / "forecasting",
    ROOT / "results" / "decision",
    ROOT / "src" / "data",
    ROOT / "src" / "features",
    ROOT / "src" / "clustering",
    ROOT / "src" / "forecasting",
    ROOT / "src" / "decision",
    ROOT / "src" / "utils",
]


def ensure_directories():
    for folder in TARGET_DIRS:
        folder.mkdir(parents=True, exist_ok=True)
    print("Created/verified directory structure.")



def safe_move(src: Path, dst: Path):
    if not src.exists():
        print(f"[SKIP] Source not found: {src}")
        return

    if dst.exists():
        print(f"[SKIP] Target already exists: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"[MOVED] {src} -> {dst}")



def move_root_scripts():
    file_map = {
        ROOT / "data_loader.py": ROOT / "src" / "data" / "data_loader.py",
        ROOT / "run_forecasting.py": ROOT / "src" / "forecasting" / "run_forecasting.py",
        ROOT / "train_lightgbm.py": ROOT / "src" / "forecasting" / "train_lightgbm.py",
        ROOT / "calc_final_inventory.py": ROOT / "src" / "decision" / "calc_final_inventory.py",
        ROOT / "test_inventory.py": ROOT / "src" / "decision" / "test_inventory.py",
    }

    for src, dst in file_map.items():
        safe_move(src, dst)



def move_result_files():
    file_map = {
        ROOT / "results" / "cluster_assignments_k3.csv": ROOT / "results" / "clustering" / "cluster_assignments_k3.csv",
        ROOT / "results" / "cluster_assignments_k4.csv": ROOT / "results" / "clustering" / "cluster_assignments_k4.csv",
        ROOT / "results" / "cluster_assignments_k5.csv": ROOT / "results" / "clustering" / "cluster_assignments_k5.csv",
        ROOT / "results" / "cluster_assignments_k6.csv": ROOT / "results" / "clustering" / "cluster_assignments_k6.csv",
        ROOT / "results" / "cluster_assignments_k7.csv": ROOT / "results" / "clustering" / "cluster_assignments_k7.csv",
        ROOT / "results" / "cluster_assignments_k8.csv": ROOT / "results" / "clustering" / "cluster_assignments_k8.csv",
        ROOT / "results" / "cluster_pattern_summary_k7.csv": ROOT / "results" / "clustering" / "cluster_pattern_summary_k7.csv",
        ROOT / "results" / "cluster_patterns_k7.png": ROOT / "results" / "clustering" / "cluster_patterns_k7.png",
        ROOT / "results" / "fused_clustering_metrics.csv": ROOT / "results" / "clustering" / "fused_clustering_metrics.csv",

        ROOT / "results" / "inventory_decision_k7.csv": ROOT / "results" / "decision" / "inventory_decision_k7.csv",
    }

    for src, dst in file_map.items():
        safe_move(src, dst)



def ensure_forecasting_files():
    forecasting_files = [
        ROOT / "results" / "forecasting" / "evaluation_metrics_by_sku.csv",
        ROOT / "results" / "forecasting" / "evaluation_summary_by_cluster.csv",
        ROOT / "results" / "forecasting" / "final_comparison_with_lgbm.csv",
        ROOT / "results" / "forecasting" / "lgbm_evaluation_by_sku.csv",
    ]

    for file in forecasting_files:
        if file.exists():
            print(f"[OK] Forecasting file already in place: {file}")
        else:
            print(f"[WARN] Forecasting file not found: {file}")



def print_summary():
    print("\nSuggested final structure:")
    print(
        """
M5-CENSORED-DEMAND-PATTERNS/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
├─ results/
│  ├─ clustering/
│  ├─ forecasting/
│  └─ decision/
├─ src/
│  ├─ data/
│  ├─ features/
│  ├─ clustering/
│  ├─ forecasting/
│  ├─ decision/
│  └─ utils/
├─ main.py
├─ README.md
├─ requirements.txt
└─ reorganize_project.py
"""
    )


def main():
    print("Starting project reorganization...\n")
    ensure_directories()
    move_root_scripts()
    move_result_files()
    ensure_forecasting_files()
    print_summary()
    print("Done.")


if __name__ == "__main__":
    main()