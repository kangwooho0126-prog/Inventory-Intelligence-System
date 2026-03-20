import numpy as np


pattern_policy = {
    "stable": {
        "safety_factor": 1.2,
        "description": "Stable demand → lower inventory"
    },
    "intermittent": {
        "safety_factor": 1.5,
        "description": "Intermittent demand → moderate inventory"
    },
    "censored": {
        "safety_factor": 2.0,
        "description": "Censored demand → higher inventory"
    }
}


def compute_safety_stock(std, lead_time, z=1.65):
    """
    Calculate basic safety stock
    """
    return z * std * np.sqrt(lead_time)



def adjusted_safety_stock(std, lead_time, pattern):
    base_ss = compute_safety_stock(std, lead_time)

    factor = pattern_policy.get(pattern, {}).get("safety_factor", 1.0)
    return base_ss * factor


def reorder_point(mean_demand, lead_time, safety_stock):
    return mean_demand * lead_time + safety_stock



def inventory_decision(mean_demand, std, lead_time, pattern):
    ss = adjusted_safety_stock(std, lead_time, pattern)
    rop = reorder_point(mean_demand, lead_time, ss)

    return {
        "pattern": pattern,
        "safety_stock": float(ss),
        "reorder_point": float(rop)
    }