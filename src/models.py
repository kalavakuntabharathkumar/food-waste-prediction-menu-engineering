import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

FEATURES = ["cafeteria","dish","dow","month","is_weekend","rolling_7","rolling_28"]

def train_xgb(train):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["cafeteria","dish"]),
        ("num", "passthrough", ["dow","month","is_weekend","rolling_7","rolling_28"])
    ])
    model = xgb.XGBRegressor(
        n_estimators=250, max_depth=7, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85, objective="reg:squarederror",
        random_state=42
    )
    pipe = Pipeline([("prep", pre), ("model", model)])
    pipe.fit(train[FEATURES], train["meals_sold"])
    return pipe
