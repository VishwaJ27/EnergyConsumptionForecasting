import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from preprocess import load_data

def run_sarima(df):
    # split data - last 30 days for testing
    train = df["y"][:-30]
    test = df["y"][-30:]

    # fit SARIMA model
    # (1,1,1) is a simple starting order - works well for most energy data
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    result = model.fit(disp=False)

    # forecast next 30 days
    forecast = result.forecast(steps=30)
    forecast_index = df["ds"][-30:].values

    # plot the forecast
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["ds"], df["y"], label="Actual", color="blue")
    ax.plot(forecast_index, forecast.values, label="Forecast", color="green")
    ax.set_title("SARIMA - Energy Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("sarima_forecast.png")
    plt.show()
    print("SARIMA plot saved!")

    return forecast, test


if __name__ == "__main__":
    df = load_data("data/household_power_consumption.csv")
    forecast, test = run_sarima(df)
    print(forecast)