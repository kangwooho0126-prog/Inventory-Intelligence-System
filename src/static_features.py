import numpy as np
import pandas as pd


def safe_divide(a, b):
    """
    Safely divide a by b. Return 0.0 if denominator is 0.
    """
    return float(a / b) if b != 0 else 0.0


def calculate_mean_sales(series: np.ndarray) -> float:
    return float(np.mean(series))


def calculate_zero_ratio(series: np.ndarray) -> float:
    return float(np.mean(series == 0))


def calculate_std_sales(series: np.ndarray) -> float:
    return float(np.std(series, ddof=0))


def calculate_cv(series: np.ndarray) -> float:
    mean_sales = calculate_mean_sales(series)
    std_sales = calculate_std_sales(series)
    return safe_divide(std_sales, mean_sales)


def calculate_slope(series: np.ndarray) -> float:
    """
    Linear trend slope using least squares.
    """
    x = np.arange(len(series))
    if len(series) < 2:
        return 0.0
    slope = np.polyfit(x, series, 1)[0]
    return float(slope)


def calculate_tail_zero_days(series: np.ndarray) -> int:
    """
    Number of consecutive zero-sales days at the end of the series.
    """
    count = 0
    for value in reversed(series):
        if value == 0:
            count += 1
        else:
            break
    return int(count)


def calculate_gini(series: np.ndarray) -> float:
    """
    Gini coefficient of the sales distribution.
    """
    series = np.asarray(series, dtype=float)

    if np.all(series == 0):
        return 0.0

    if np.any(series < 0):
        series = series - np.min(series)

    series = np.sort(series)
    n = len(series)
    index = np.arange(1, n + 1)

    numerator = np.sum((2 * index - n - 1) * series)
    denominator = n * np.sum(series)

    return safe_divide(numerator, denominator)


def calculate_seasonality_strength(series: np.ndarray, period: int = 7) -> float:
    """
    Simple seasonality strength:
    variance of seasonal means / total variance
    Default period=7 for weekly seasonality in daily sales.
    """
    series = np.asarray(series, dtype=float)

    if len(series) < period or np.var(series) == 0:
        return 0.0

    seasonal_groups = [[] for _ in range(period)]
    for i, value in enumerate(series):
        seasonal_groups[i % period].append(value)

    seasonal_means = np.array([
        np.mean(group) if len(group) > 0 else 0.0
        for group in seasonal_groups
    ])

    strength = np.var(seasonal_means) / np.var(series)
    return float(strength)


def calculate_active_span(series: np.ndarray) -> int:
    """
    Distance (inclusive) between first non-zero sale and last non-zero sale.
    If all zero, return 0.
    """
    nonzero_idx = np.where(series > 0)[0]
    if len(nonzero_idx) == 0:
        return 0
    return int(nonzero_idx[-1] - nonzero_idx[0] + 1)


def calculate_nonzero_runs(series: np.ndarray) -> int:
    """
    Number of contiguous non-zero sales segments.
    Example: [0,2,3,0,1,1,0] => 2
    """
    count = 0
    in_run = False

    for value in series:
        if value > 0 and not in_run:
            count += 1
            in_run = True
        elif value == 0:
            in_run = False

    return int(count)


def calculate_sales_burst_ratio(series: np.ndarray) -> float:
    """
    Peak sales / mean non-zero sales.
    Measures burstiness of demand.
    """
    nonzero_values = series[series > 0]
    if len(nonzero_values) == 0:
        return 0.0

    peak = np.max(nonzero_values)
    mean_nonzero = np.mean(nonzero_values)

    return safe_divide(peak, mean_nonzero)


def calculate_max_zero_run(series: np.ndarray) -> int:
    """
    Longest consecutive zero-sales run.
    """
    max_run = 0
    current_run = 0

    for value in series:
        if value == 0:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return int(max_run)


def extract_static_features_for_series(series: np.ndarray, item_id: str = None) -> dict:
    """
    Extract the 12 static features from one sales series.
    """
    series = np.asarray(series, dtype=float)

    features = {
        "item_id": item_id,
        "mean_sales": calculate_mean_sales(series),
        "zero_ratio": calculate_zero_ratio(series),
        "cv": calculate_cv(series),
        "slope": calculate_slope(series),
        "tail_zero_days": calculate_tail_zero_days(series),
        "gini": calculate_gini(series),
        "seasonality_strength": calculate_seasonality_strength(series, period=7),
        "active_span": calculate_active_span(series),
        "nonzero_runs": calculate_nonzero_runs(series),
        "std_sales": calculate_std_sales(series),
        "sales_burst_ratio": calculate_sales_burst_ratio(series),
        "max_zero_run": calculate_max_zero_run(series),
    }

    return features


def extract_static_features_from_dataframe(
    df: pd.DataFrame,
    item_id_col: str = "item_id"
) -> pd.DataFrame:
    """
    Extract static features for all items in a dataframe.

    Expected format:
    - one row per item
    - first column can be item_id
    - remaining columns are daily sales values

    Example:
        item_id, d_1, d_2, d_3, ..., d_365
    """
    feature_rows = []

    sales_columns = [col for col in df.columns if col != item_id_col]

    for _, row in df.iterrows():
        item_id = row[item_id_col] if item_id_col in df.columns else None
        series = row[sales_columns].values.astype(float)

        feature_row = extract_static_features_for_series(series, item_id=item_id)
        feature_rows.append(feature_row)

    return pd.DataFrame(feature_rows)