import pandas as pd
import numpy as np

def menu_metrics(df):
    g = df.groupby("dish").agg(
        sold=("meals_sold","sum"),
        produced=("meals_produced","sum"),
        avg_daily_sold=("meals_sold","mean")
    ).reset_index()
    g["waste_units"] = (g["produced"] - g["sold"]).clip(lower=0)
    g["waste_rate"] = g["waste_units"] / g["produced"].replace(0, np.nan)
    g["velocity_rank"] = g["sold"].rank(pct=True)
    g["waste_rank"] = g["waste_rate"].rank(pct=True)
    g["recommendation"] = np.select(
        [(g.velocity_rank >= .75) & (g.waste_rate < .10),
         (g.waste_rate >= .20),
         (g.velocity_rank < .35)],
        ["Keep / prioritize", "Reduce production", "Review / rotate"],
        default="Maintain"
    )
    return g.fillna(0)

def estimate_savings(df, unit_cost=35, reduction=0.22):
    waste = max(0, (df["meals_produced"]-df["meals_sold"]).sum())
    return waste * reduction * unit_cost
