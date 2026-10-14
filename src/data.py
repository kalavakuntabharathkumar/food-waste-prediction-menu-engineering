import numpy as np
import pandas as pd

def generate_meal_data(n_days=420, cafeterias=6, dishes=180, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        dow = d.dayofweek
        season = 1 + 0.10*np.sin(2*np.pi*d.dayofyear/365)
        for c in range(1, cafeterias+1):
            for dish in rng.choice(np.arange(1, dishes+1), size=12, replace=False):
                base = 70 + 18*np.sin(dish/13) + 12*(dow < 5)
                demand = max(5, rng.normal(base*season, 10))
                produced = max(demand, rng.normal(demand*1.10, 7))
                rows.append((d, f"C{c}", f"D{dish:03d}", round(demand), round(produced)))
    return pd.DataFrame(rows, columns=["date","cafeteria","dish","meals_sold","meals_produced"])

def load_data(path=None):
    if path and __import__("os").path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return generate_meal_data()
