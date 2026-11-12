"""Streamlit dashboard: 24-hour grid demand forecast with peak hours + capacity warning."""
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))
from features import forecast  # noqa: E402

st.set_page_config(page_title="Grid Demand Forecast", page_icon="⚡", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(APP_DIR / "model.joblib")


try:
    bundle = load_bundle()
except FileNotFoundError:
    st.error("`model.joblib` not found. Run the notebook (Step 9) to create it.")
    st.stop()

model, history = bundle["model"], bundle["history"]
first_day = history.index[0].normalize() + pd.Timedelta(days=8)   # need 7+ days of lag history
last_day = history.index[-1].normalize()                          # last day with real data
max_day = last_day + pd.Timedelta(days=7)

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Forecast settings")
picked = st.sidebar.date_input("Forecast date", value=last_day + pd.Timedelta(days=1),
                               min_value=first_day.date(), max_value=max_day.date())
capacity = st.sidebar.number_input("Grid capacity limit (MW)", min_value=1000, max_value=100000,
                                   value=45000, step=500)
n_peaks = st.sidebar.slider("Peak hours to highlight", 1, 6, 3)
st.sidebar.caption(f"Model: **{bundle['model_name']}**  \n"
                   f"Test MAPE: **{bundle['metrics']['MAPE']:.2f}%**, RMSE: **{bundle['metrics']['RMSE']:.0f} MW**")

# ------------------------------------------------------------------ title
st.title("⚡ Electricity Demand Forecast — PJM East")
st.caption("Day-ahead hourly demand forecast to help grid operators avoid under- and over-forecasting.")

if isinstance(picked, (tuple, list)):            # user cleared / partially selected the date
    if not picked:
        st.warning("Please pick a date.")
        st.stop()
    picked = picked[0]
day = pd.Timestamp(picked)

# ------------------------------------------------------------------ forecast
try:
    pred = forecast(model, history, day)
except Exception as exc:                          # never crash on bad input
    st.error(f"Could not build a forecast for {day.date()}: {exc}")
    st.stop()

if day > last_day + pd.Timedelta(days=1):
    st.info(f"Data ends on {last_day.date()}. This date is forecast recursively from earlier "
            "predictions, so uncertainty grows with every extra day.")

actual = history.reindex(pred.index) if day <= last_day else None
peak_idx = pred.nlargest(n_peaks).index
over = pred[pred > capacity]

# ------------------------------------------------------------------ KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Peak demand", f"{pred.max():,.0f} MW")
c2.metric("Peak hour", f"{pred.idxmax():%H:%M}")
c3.metric("Average demand", f"{pred.mean():,.0f} MW")
c4.metric("Hours above limit", f"{len(over)}")

# ------------------------------------------------------------------ alert
if len(over):
    hours = ", ".join(f"{t:%H:%M}" for t in over.index)
    st.error(f"🚨 **Capacity warning:** forecast demand exceeds {capacity:,.0f} MW at {hours}. "
             f"Worst case: {pred.max() - capacity:,.0f} MW above the limit. "
             "Consider scheduling reserve generation or demand response.")
else:
    st.success(f"✅ Forecast stays below {capacity:,.0f} MW all day "
               f"(headroom at peak: {capacity - pred.max():,.0f} MW).")

# ------------------------------------------------------------------ chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred.index, y=pred.values, mode="lines+markers", name="Forecast",
                         line=dict(color="#2563eb", width=3)))
fig.add_trace(go.Scatter(x=peak_idx, y=pred[peak_idx], mode="markers", name=f"Top {n_peaks} peak hours",
                         marker=dict(color="#dc2626", size=14, symbol="star")))
if actual is not None:
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, mode="lines", name="Actual (backtest)",
                             line=dict(color="gray", dash="dot")))
fig.add_hline(y=capacity, line_dash="dash", line_color="#f59e0b",
              annotation_text=f"Capacity {capacity:,.0f} MW")
fig.update_layout(height=480, xaxis_title="Hour", yaxis_title="Demand (MW)",
                  margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

if actual is not None and actual.notna().all():
    mape = ((actual - pred).abs() / actual).mean() * 100
    st.caption(f"Backtest for this day: MAPE = {mape:.2f}%  (the deployed model was refit on all data, "
               "so this backtest is optimistic; see the notebook for honest test scores).")

with st.expander("Show forecast table"):
    table = pred.rename("Forecast (MW)").round(0).to_frame()
    table["Peak hour"] = table.index.isin(peak_idx)
    table["Above limit"] = table["Forecast (MW)"] > capacity
    st.dataframe(table)
    st.download_button("Download CSV", table.to_csv().encode(), file_name=f"forecast_{day.date()}.csv")

st.markdown("---")
st.caption("Limitations: trained on historical PJM East data (2002–2018); no weather inputs; "
           "extreme events (heat waves, storms, pandemics) can cause larger errors. "
           "Data: Kaggle robikscube/hourly-energy-consumption.")
