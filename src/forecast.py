from prophet import Prophet

def prophet_forecast(df, periods=28):
    daily = df.groupby("date", as_index=False)["meals_sold"].sum()
    daily = daily.rename(columns={"date":"ds", "meals_sold":"y"})
    model = Prophet(weekly_seasonality=True, daily_seasonality=False)
    model.fit(daily)
    future = model.make_future_dataframe(periods=periods)
    return model, model.predict(future)
