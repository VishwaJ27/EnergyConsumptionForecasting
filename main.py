import sys
import os

# so python can find the src folder
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from preprocess import load_data
from prophet_model import run_prophet
from sarima_model import run_sarima
from evaluate import get_metrics
import numpy as np

def main():
    print("Energy Consumption Forecasting")
    print("=" * 40)

    # step 1 - load and clean data
    print("\n Loading data...")
    df = load_data("data/household_power_consumption.txt")
    print(f"   Data loaded! Total days: {len(df)}")

    # step 2 - run prophet
    print("\n Running Prophet model...")
    prophet_forecast = run_prophet(df)

    # step 3 - run sarima
    print("\n Running SARIMA model...")
    sarima_forecast, _ = run_sarima(df)

    # step 4 - evaluate both
    print("\nEvaluating models...")
    actual       = df["y"][-30:].values
    prophet_pred = prophet_forecast["yhat"][-30:].values
    sarima_pred  = sarima_forecast.values

    get_metrics(actual, prophet_pred, "Prophet")
    get_metrics(actual, sarima_pred,  "SARIMA")

    print("\n Done! Check prophet_forecast.png and sarima_forecast.png")
    print("=" * 40)


if __name__ == "__main__":
    main()