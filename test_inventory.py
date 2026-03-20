from src.inventory_decision import inventory_decision

result = inventory_decision(
    mean_demand=50,
    std=10,
    lead_time=5,
    pattern="censored"
)

print(result)