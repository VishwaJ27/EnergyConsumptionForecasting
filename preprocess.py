import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath, sep=";", low_memory=False)
    
    # combine date and time into one column
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True)
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime")

    # only keep the main power column
    df = df[["Global_active_power"]]

    # replace missing values marked as '?' 
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.fillna(method="ffill")

    # resample to daily average
    df = df.resample("D").mean()
    df.columns = ["y"]
    df.index.name = "ds"
    df = df.reset_index()

    return df


if __name__ == "__main__":
    df = load_data("data/household_power_consumption.csv")
    print(df.head())
    print(df.shape)