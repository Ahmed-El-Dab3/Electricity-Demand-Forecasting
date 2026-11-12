# ⚡ Electricity Demand Forecasting for the Power Grid

> Grid operators lose money (and risk blackouts) when tomorrow's demand is forecast badly — this project predicts **hourly electricity demand (MW) day-ahead** and warns when demand will exceed grid capacity.

🔗 **Live app:** `<paste your Streamlit Community Cloud link>`

![App screenshot](presentation/app_screenshot.png)

## Results (untouched test set: 2017 → Aug 2018)

| Model | MAPE % | RMSE (MW) |
|---|---|---|
| Baseline — same hour yesterday (`lag_24`) | [X] | [X] |
| Ridge Regression | [X] | [X] |
| Random Forest | [X] | [X] |
| **LightGBM (best)** | **[X]** | **[X]** |

_Fill the table from the notebook's Step 7 output — never write a number you did not measure._

## What's inside
| Path | Purpose |
|---|---|
| `notebook.ipynb` | Full 9-step workflow (framing → EDA → features → modeling → tuning → evaluation → insights → export) |
| `app/features.py` | Feature engineering shared by notebook and app (guarantees identical features) |
| `app/app.py` | Streamlit dashboard |
| `app/model.joblib` | Saved model + demand history (created by notebook Step 9) |
| `presentation/` | Slide outline and LinkedIn post template |

## Approach
* **Features:** calendar (hour, weekday, month, year, quarter, day-of-year, weekend, US holiday), cyclical hour (sin/cos), lags (24 h, 48 h, 168 h), 24 h rolling mean/std (shifted so they are known at day-ahead time).
* **No leakage:** chronological split (train ≤ 2016, test ≥ 2017); lags/rolling use past data only; scaler inside the sklearn `Pipeline`.
* **Tuning:** `GridSearchCV` / `RandomizedSearchCV` with `TimeSeriesSplit` on the training set only.
* **App:** pick a date → 24 h forecast, peak hours highlighted, capacity-limit warning, CSV download.

## Run it
```bash
pip install -r requirements.txt
# 1. download PJME_hourly.csv from Kaggle into data/raw/
# 2. run all cells of notebook.ipynb  (creates app/model.joblib)
streamlit run app/app.py
```

## Limitations
No weather inputs, data ends in 2018, point forecasts only, multi-day forecasts are recursive.

## Data
PJM Hourly Energy Consumption — https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption (file `PJME_hourly.csv`). Not included in this repo; please download it from Kaggle.

## CV bullet
Forecast hourly electricity demand on 10+ years of real grid data (MAPE [X]%) and deployed a peak-load warning dashboard for grid planning (Python, scikit-learn, LightGBM, Streamlit).
