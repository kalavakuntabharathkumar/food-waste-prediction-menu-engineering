# Food Waste Prediction & Menu Engineering Analytics

A portfolio-ready analytics project for forecasting meal demand, estimating waste risk,
explaining model drivers, and exploring menu-engineering recommendations.

## Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Prophet, Statsmodels, SHAP, MLflow, Streamlit.

## Project structure
- `app.py` — Streamlit dashboard
- `src/data.py` — synthetic meal-level data generator and loading helpers
- `src/features.py` — feature engineering
- `src/models.py` — XGBoost demand model
- `src/forecast.py` — Prophet forecast helper
- `src/menu_engineering.py` — menu metrics and waste-reduction recommendations
- `src/evaluate.py` — MAPE/RMSE evaluation
- `scripts/train.py` — reproducible training entry point
- `tests/` — unit tests
- `mlruns/` — created automatically when MLflow tracking is used

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/train.py
streamlit run app.py
```

The repository uses generated/synthetic data so it is immediately runnable without exposing
any real cafeteria records. The dashboard labels the data as synthetic.
