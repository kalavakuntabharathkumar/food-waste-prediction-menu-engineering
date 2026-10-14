import pandas as pd
import numpy as np

def add_features(df):
    x = df.copy().sort_values("date")
    x["dow"] = x["date"].dt.dayofweek
    x["month"] = x["date"].dt.month
    x["is_weekend"] = (x["dow"] >= 5).astype(int)
    x["rolling_7"] = x.groupby(["cafeteria","dish"])["meals_sold"].transform(
        lambda s: s.shift(1).rolling(7, min_periods=2).mean()
    )
    x["rolling_28"] = x.groupby(["cafeteria","dish"])["meals_sold"].transform(
        lambda s: s.shift(1).rolling(28, min_periods=5).mean()
    )
    x["waste_units"] = (x["meals_produced"] - x["meals_sold"]).clip(lower=0)
    x["waste_pct"] = x["waste_units"] / x["meals_produced"].replace(0, np.nan)
    return x.fillna({"rolling_7":0, "rolling_28":0, "waste_pct":0})
