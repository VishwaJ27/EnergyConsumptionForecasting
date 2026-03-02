# Energy Consumption Forecasting

A time series forecasting project that predicts household energy usage using Prophet and SARIMA models, with an interactive Streamlit dashboard for visualization.

---

## Overview

This project loads historical household electric power data, cleans it, and trains two forecasting models to predict the next 30 days of energy consumption. The results are displayed in a professional dark-themed dashboard where both models can be compared side by side.

---

## Project Structure

```
energy-forecasting/
├── data/
│   └── household_power_consumption.txt    # Raw dataset
├── notebooks/
│   └── eda.ipynb                          # Exploratory analysis
├── preprocess.py                          # Load and clean data
├── prophet_model.py                       # Prophet forecast
├── sarima_model.py                        # SARIMA forecast
├── evaluate.py                            # Model comparison metrics
├── dashboard.py                           # Streamlit dashboard
├── main.py                                # Run full pipeline
└── requirements.txt                       # Dependencies
```

---

## Dataset

Download the dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

Place the file inside the `data/` folder. The file uses `;` as a separator and contains minute-level power readings which are resampled to daily averages during preprocessing.

---

## Installation

**Step 1 — Clone or download the project**

**Step 2 — Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 3 — Place the dataset in the data folder**

---

## Usage

**Run the full pipeline (terminal output)**

```bash
python main.py
```

**Run the interactive dashboard**

```bash
streamlit run dashboard.py
```

---

## Models

| Model | Description |
|---|---|
| Prophet | Facebook's forecasting library, handles seasonality and trends automatically |
| SARIMA | Statistical model from Statsmodels, captures weekly seasonal patterns |

Both models are trained on all data except the last 30 days, which are used for evaluation.

---

## Dashboard Features

- Toggle between Prophet, SARIMA, or both models
- Adjust forecast window from 7 to 60 days
- View MAE and RMSE metrics for each model
- Automatic winner selection based on MAE
- Raw data table toggle

---

## Evaluation Metrics

- **MAE** — Mean Absolute Error, average difference between actual and predicted values
- **RMSE** — Root Mean Squared Error, penalizes larger errors more heavily

---

## Requirements

```
pandas
numpy
matplotlib
prophet
statsmodels
scikit-learn
streamlit
jupyter
```

---

## License

This project is for educational purposes. The dataset is provided by UCI Machine Learning Repository.
