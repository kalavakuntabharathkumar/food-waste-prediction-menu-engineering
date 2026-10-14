import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import mlflow
from src.data import load_data
from src.features import add_features
from src.models import train_xgb
from src.evaluate import mape, rmse

df = add_features(load_data())
cut = df["date"].quantile(.8)
train, test = df[df.date <= cut], df[df.date > cut]
with mlflow.start_run():
    model = train_xgb(train)
    pred = model.predict(test[["cafeteria","dish","dow","month","is_weekend","rolling_7","rolling_28"]])
    mlflow.log_metric("test_mape", mape(test.meals_sold, pred))
    mlflow.log_metric("test_rmse", rmse(test.meals_sold, pred))
    mlflow.sklearn.log_model(model, "xgboost_model")
    print("MAPE:", mape(test.meals_sold, pred))
    print("RMSE:", rmse(test.meals_sold, pred))
