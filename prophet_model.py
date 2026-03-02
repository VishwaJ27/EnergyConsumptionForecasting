import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from preprocess import load_data

def run_prophet(df):
    # split data - last 30 days for testing
    train = df[:-30]
    test = df[-30:]

    # fit model
    model = Prophet()
    model.fit(train)

    # forecast next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # plot the forecast
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["ds"], df["y"], label="Actual", color="blue")
    ax.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="orange")
    ax.set_title("Prophet - Energy Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Power (kW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("prophet_forecast.png")
    plt.show()
    print("Prophet plot saved!")

    return forecast


if __name__ == "__main__":
    df = load_data("data/household_power_consumption.csv")
    forecast = run_prophet(df)
    print(forecast[["ds", "yhat"]].tail(30))