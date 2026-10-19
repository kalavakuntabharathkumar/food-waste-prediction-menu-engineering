from src.data import generate_meal_data
from src.features import add_features
from src.menu_engineering import menu_metrics

def test_generated_data():
    df = generate_meal_data(n_days=10, cafeterias=2, dishes=20)
    assert len(df) > 0
    assert {"date","cafeteria","dish","meals_sold","meals_produced"} <= set(df.columns)

def test_features_and_menu_metrics():
    df = add_features(generate_meal_data(n_days=20, cafeterias=2, dishes=20))
    m = menu_metrics(df)
    assert "waste_rate" in m.columns
    assert (df.waste_units >= 0).all()
