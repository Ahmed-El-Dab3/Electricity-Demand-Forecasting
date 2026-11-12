"""Shared feature engineering + forecasting helpers (used by the notebook AND the Streamlit app).

Design rule: every feature is available at DAY-AHEAD time, i.e. at 23:00 of the previous day.
That is why the rolling statistics are computed on the series shifted by 24 hours.
"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

TARGET = "PJME_MW"

FEATURES = [
    "hour", "dayofweek", "month", "year", "quarter", "dayofyear",
    "is_weekend", "is_holiday",
    "sin_hour", "cos_hour",
    "lag_24", "lag_48", "lag_168",
    "rolling_mean_24", "rolling_std_24",
]


def make_features(s: pd.Series) -> pd.DataFrame:
    """Build calendar, cyclical, lag and rolling features from an hourly demand Series.

    Only PAST values are used (shift >= 24), so no information from the future leaks in.
    Rows where the target is NaN are allowed (they are the hours we want to forecast).
    """
    df = pd.DataFrame({TARGET: s})
    idx = df.index

    # calendar features
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["year"] = idx.year
    df["quarter"] = idx.quarter
    df["dayofyear"] = idx.dayofyear
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(idx.min().normalize(), idx.max().normalize())
    df["is_holiday"] = idx.normalize().isin(holidays).astype(int)

    # cyclical encoding of the hour (23:00 and 00:00 are neighbours)
    df["sin_hour"] = np.sin(2 * np.pi * idx.hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * idx.hour / 24)

    # lags: same hour yesterday / 2 days ago / last week
    df["lag_24"] = s.shift(24)
    df["lag_48"] = s.shift(48)
    df["lag_168"] = s.shift(168)

    # rolling statistics over the 24 hours that END 24h before the target hour
    lagged = s.shift(24)
    df["rolling_mean_24"] = lagged.rolling(24).mean()
    df["rolling_std_24"] = lagged.rolling(24).std()
    return df


def forecast_day(model, history: pd.Series, day) -> pd.Series:
    """Forecast the 24 hours of `day`. `history` must be hourly, contiguous and end at 23:00 of day-1."""
    start = pd.Timestamp(day).normalize()
    hist = history[history.index < start].iloc[-24 * 10:]          # 10 days is plenty (max lag = 7 days)
    new_idx = pd.date_range(start, periods=24, freq="h")
    extended = pd.concat([hist, pd.Series(np.nan, index=new_idx)])
    X = make_features(extended).loc[new_idx, FEATURES]
    return pd.Series(model.predict(X), index=new_idx, name="forecast")


def forecast(model, history: pd.Series, day) -> pd.Series:
    """Forecast `day`. If it is more than one day after the end of the history, forecast
    recursively day by day, feeding predictions back in as pseudo-history."""
    target = pd.Timestamp(day).normalize()
    hist = history[history.index < target]
    next_day = hist.index[-1].normalize() + pd.Timedelta(days=1)
    pred = None
    while next_day <= target:
        pred = forecast_day(model, hist, next_day)
        hist = pd.concat([hist, pred.rename(TARGET)])
        next_day += pd.Timedelta(days=1)
    return pred
