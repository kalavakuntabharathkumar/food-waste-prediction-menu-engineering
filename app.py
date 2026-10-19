import streamlit as st
import pandas as pd
from src.data import load_data
from src.features import add_features
from src.menu_engineering import menu_metrics, estimate_savings
from src.forecast import prophet_forecast

st.set_page_config(page_title="Food Waste Analytics", layout="wide")
st.title("Food Waste Prediction & Menu Engineering Analytics")
st.caption("Portfolio demo using reproducible synthetic cafeteria data.")

df = add_features(load_data())
c1,c2,c3,c4 = st.columns(4)
c1.metric("Meal records", f"{len(df):,}")
c2.metric("Cafeterias", df.cafeteria.nunique())
c3.metric("Dishes", df.dish.nunique())
c4.metric("Estimated savings", f"₹{estimate_savings(df):,.0f}")

st.subheader("Waste by cafeteria")
w = df.groupby("cafeteria", as_index=False).agg(
    produced=("meals_produced","sum"), sold=("meals_sold","sum"))
w["waste"] = w.produced-w.sold
st.bar_chart(w.set_index("cafeteria")["waste"])

st.subheader("Menu engineering recommendations")
m = menu_metrics(df)
st.dataframe(m.sort_values("waste_rate", ascending=False).head(25), use_container_width=True)

if st.button("Run 28-day aggregate forecast"):
    model, forecast = prophet_forecast(df)
    st.line_chart(forecast.set_index("ds")["yhat"].tail(90))
