import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocess import load_data
from prophet_model import run_prophet
from sarima_model import run_sarima

def get_metrics(actual, predicted, model_name):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(f"\n {model_name} Results:")
    print(f"   MAE  : {mae:.4f}")
    print(f"   RMSE : {rmse:.4f}")

    return mae, rmse


if __name__ == "__main__":
    df = load_data("data/household_power_consumption.csv")

    # get actual last 30 days
    actual = df["y"][-30:].values

    # prophet predictions
    prophet_forecast = run_prophet(df)
    prophet_pred = prophet_forecast["yhat"][-30:].values

    # sarima predictions
    sarima_pred, _ = run_sarima(df)
    sarima_pred = sarima_pred.values

    # compare both models
    get_metrics(actual, prophet_pred, "Prophet")
    get_metrics(actual, sarima_pred, "SARIMA")

    # simple winner check
    prophet_mae = mean_absolute_error(actual, prophet_pred)
    sarima_mae  = mean_absolute_error(actual, sarima_pred)

    print("\n Better Model:", "Prophet" if prophet_mae < sarima_mae else "SARIMA")