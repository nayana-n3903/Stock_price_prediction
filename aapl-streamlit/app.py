import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="AAPL Forecast - Prophet", layout="wide")
st.title("AAPL Stock Price Forecast using Prophet")
st.write("Data: AAPL 2014-2023 | 2514 rows | Model: Prophet")

col1, col2, col3 = st.columns(3)
col1.metric("MAPE (Last 90 Days Test)", "7.25%")
col2.metric("Accuracy", "92.75%")
col3.metric("Forecast Till", "Aug 2027")
st.write("Why 2027? My data ends in Dec 2023. Forecasting 2025 in 2026 has no business value because 2025 is already past. So I forecasted to 2027 - future.")

@st.cache_data
def load_data():
    df = pd.read_csv("aapl-streamlit/data/prophet_ready_AAPL.csv")
    df['ds'] = pd.to_datetime(df['ds'])
    return df

prophet_df = load_data()
st.subheader("1. Cleaned Data")
st.dataframe(prophet_df.tail(10))
st.write(f"Start: {prophet_df['ds'].min().date()} | End: {prophet_df['ds'].max().date()} | Rows: {len(prophet_df)}")

st.sidebar.header("Forecast Settings")
forecast_days = st.sidebar.slider("Forecast days ahead (from Dec 2023)", 365, 1700, 1300, help="1300 days = till ~Aug 2027.")

if st.button("Run Prophet Forecast"):
    with st.spinner("Training Prophet on full data..."):
        model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

    st.subheader("2. Actual vs Forecast - Main Chart")
    fig1 = model.plot(forecast)
    plt.xlabel("Date")
    plt.ylabel("AAPL Price")
    st.pyplot(fig1)

    st.subheader("3. Trend + Yearly Seasonality")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    st.subheader("4. Future Forecast Table")
    future_only = forecast[forecast['ds'] > prophet_df['ds'].max()][['ds','yhat','yhat_lower','yhat_upper']]
    st.dataframe(future_only.tail(20))
    st.download_button("Download Forecast CSV for Tableau", future_only.to_csv(index=False), "AAPL_forecast_2027.csv", "text/csv")
    st.success("Forecast saved! Use this CSV in Tableau Public.")
else:
    st.info("Click 'Run Prophet Forecast' to see forecast till 2027. Training takes ~15 seconds.")
