# AAPL Stock Price Forecasting using Prophet

> End-to-end time-series project: **DuckDB SQL for 20 KPIs → Pandas for Cleaning + Feature Engineering → Prophet for Forecasting → Streamlit + Tableau for Deployment**

**Ticker:** AAPL | **Data:** 2014-01-03 to 2023-12-28 | **Rows:** 2,514 | **Model:** Prophet | **Test:** Last 90 Days | **MAPE:** 7.25% | **Accuracy:** 92.75%

---

## 🔗 Live Links

| Platform | Link |
| :--- | :--- |
| **Streamlit App (Interactive)** | https://stockpriceprediction-...streamlit.app <!-- REPLACE with your exact Streamlit Cloud link --> |
| **Tableau Public Dashboard** | https://public.tableau.com/... <!-- REPLACE with your exact Tableau Public link --> |
| **GitHub Repo** | https://github.com/nayana-n3903/Stock_price_prediction |

---

## 📌 What This Project Actually Does

Most students jump straight to Prophet. I did not.

My real pipeline was:

```
StockPriceDataset.csv (9 stocks)
      ↓
[1] DuckDB SQL → Explore AAPL + Calculate 20 KPIs
      ↓
[2] Pandas → Clean Data + Feature Engineering
      ↓
[3] Prophet → Train / Test / Forecast till Aug 2027
      ↓
[4] Streamlit (Interactive) + Tableau Public (Storytelling)
```

**Why this order matters:**
- DuckDB SQL tells me *what is in the data* before I model it.
- Pandas cleaning makes the data *trustworthy*.
- Feature Engineering makes the data *useful for forecasting*.
- Prophet only works well if the first 3 steps are correct.

---

## 🗂️ Dataset

- **Source File:** `StockPriceDataset.csv`
- **Tickers in file:** AAPL, MSFT, JPM, GS, AMZN, PG, KO, XOM, CAT (9 US stocks)
- **Chosen Ticker:** `AAPL` — most liquid, cleanest trend, easiest for interviewer to understand
- **AAPL Range:** 2014-01-03 to 2023-12-28
- **Rows after cleaning:** 2,514 rows

| Column in Raw File | Meaning |
| :--- | :--- |
| `Date` | Trading date |
| `Open` | Opening price |
| `High` | Highest price of day |
| `Low` | Lowest price of day |
| `Close` | Closing price |
| `Adj Close` | Adjusted close (used for modelling — accounts for splits/dividends) |
| `Volume` | Shares traded that day |
| `Ticker` | Stock symbol |

---

## 1️⃣ Step 1: DuckDB SQL — Exploration + 20 KPIs

I loaded the CSV in DuckDB and used SQL to understand AAPL before touching Python modelling.

```sql
-- Example: Load and filter AAPL
CREATE TABLE stocks AS SELECT * FROM read_csv('StockPriceDataset.csv');
SELECT * FROM stocks WHERE Ticker = 'AAPL' ORDER BY Date;
```

### The 20 KPIs I Found Using DuckDB SQL

These are the business numbers I calculated in SQL. They told me the story of AAPL before forecasting.

#### A. Data Quality KPIs (1-4)

| # | KPI | How I Found It (SQL Logic) | What It Told Me |
| :--- | :--- | :--- | :--- |
| 1 | **Total Trading Days** | `COUNT(*) WHERE Ticker='AAPL'` | 2,514 rows — 10 years of data, enough for Prophet |
| 2 | **Date Range** | `MIN(Date), MAX(Date)` | 2014-01-03 to 2023-12-28 |
| 3 | **Missing Values** | `COUNT(*) - COUNT("Adj Close")` | 0 missing — clean price column |
| 4 | **Duplicate Dates** | `GROUP BY Date HAVING COUNT(*) > 1` | 0 duplicates after cleaning |

#### B. Price KPIs (5-10)

| # | KPI | SQL Logic | Insight |
| :--- | :--- | :--- | :--- |
| 5 | **First Adj Close (2014)** | `FIRST_VALUE("Adj Close") ORDER BY Date` | Starting price in 2014 — baseline |
| 6 | **Last Adj Close (2023)** | `LAST_VALUE("Adj Close") ORDER BY Date` | Ending price in 2023 — ~8x growth |
| 7 | **Minimum Price** | `MIN("Adj Close")` | Lowest point — crash / buying opportunity |
| 8 | **Maximum Price** | `MAX("Adj Close")` | Highest point — peak |
| 9 | **Average Price** | `AVG("Adj Close")` | Typical price level over 10 years |
| 10 | **Median Price** | `MEDIAN("Adj Close")` | Middle price — less affected by spikes than average |

#### C. Return & Risk KPIs (11-16)

| # | KPI | SQL Logic | Insight |
| :--- | :--- | :--- | :--- |
| 11 | **Total Return %** | `(Last - First) / First * 100` | Total growth over 10 years |
| 12 | **CAGR % (Yearly Growth)** | `POWER(Last/First, 1/10) - 1` | Average yearly growth — long-term trend is up |
| 13 | **Average Daily Return %** | `AVG((Close - LAG(Close)) / LAG(Close))` using `LAG() OVER (ORDER BY Date)` | Typical daily move is very small, near 0 |
| 14 | **Daily Volatility (Std Dev)** | `STDDEV(daily_return)` | How risky / jumpy AAPL is day-to-day |
| 15 | **Best Single Day %** | `MAX(daily_return)` | Biggest one-day jump |
| 16 | **Worst Single Day %** | `MIN(daily_return)` | Biggest one-day drop — risk |

#### D. Market Behaviour KPIs (17-20)

| # | KPI | SQL Logic | Insight |
| :--- | :--- | :--- | :--- |
| 17 | **% Positive Days** | `SUM(CASE WHEN Close > LAG(Close) THEN 1 ELSE 0 END) / COUNT(*) * 100` | More than 50% days are up — bullish bias |
| 18 | **% Negative Days** | `100 - % Positive Days` | Down days are fewer |
| 19 | **Average Daily Volume** | `AVG(Volume)` | Liquidity — AAPL is highly traded |
| 20 | **Average Intraday Range %** | `AVG((High - Low) / Low * 100)` | Average movement within a day — intraday volatility |

> **Interview line:** "I did not directly run Prophet. First I ran 20 KPIs in DuckDB SQL to understand price, return, risk and volume. Only after I knew AAPL was growing, volatile, and mostly bullish, I cleaned and modelled it."

---

## 2️⃣ Step 2: Pandas — Cleaning + Feature Engineering

DuckDB told me *what* the data was. Pandas made it *model-ready*.

### Cleaning in Pandas

```python
import pandas as pd

df = pd.read_csv('StockPriceDataset.csv')
aapl = df[df['Ticker'] == 'AAPL'].copy()

aapl['Date'] = pd.to_datetime(aapl['Date'])
aapl = aapl.sort_values('Date')
aapl = aapl.drop_duplicates(subset=['Date'])
aapl = aapl.dropna(subset=['Date', 'Adj Close'])
aapl = aapl.reset_index(drop=True)
```

**Cleaning rules I followed:**
1. Filter only `Ticker == 'AAPL'`
2. Convert `Date` to datetime
3. Sort by `Date` (Prophet needs sorted time-series)
4. Drop duplicate dates
5. Drop NaN in `Date` / `Adj Close`
6. Keep `Adj Close` as target — not `Close` — because it is adjusted for splits/dividends

### Feature Engineering in Pandas (+ DuckDB SQL)

I created features that help Prophet and help me explain the stock:

| Feature | Formula | Why It Matters |
| :--- | :--- | :--- |
| **Daily Return** | `(Close - Prev Close) / Prev Close` | Day-to-day growth |
| **Log Return** | `log(Close / Prev Close)` | More stable for modelling, handles compounding |
| **Intraday Range %** | `(High - Low) / Low * 100` | How much price moved *inside* the day |
| **Open-Close Spread** | `Close - Open` | Did day end up or down |
| **200-Day SMA** | `Rolling(200).mean()` on Adj Close | Long-term trend line — price above SMA = uptrend |
| **Cumulative Return** | `(Adj Close / First Price) - 1` | Total growth from start till that day |
| **Month / Quarter** | `Date.dt.month / quarter` | Seasonality check |
| **Year** | `Date.dt.year` | Yearly trend |

```python
# Example: Pandas Feature Engineering
aapl['daily_return'] = aapl['Adj Close'].pct_change()
aapl['log_return'] = np.log(aapl['Adj Close'] / aapl['Adj Close'].shift(1))
aapl['intraday_range_pct'] = (aapl['High'] - aapl['Low']) / aapl['Low'] * 100
aapl['sma_200'] = aapl['Adj Close'].rolling(200).mean()
aapl['cum_return'] = (aapl['Adj Close'] / aapl['Adj Close'].iloc[0]) - 1
```

### Final Prophet-Ready File

```python
prophet_df = aapl[['Date', 'Adj Close']].rename(columns={'Date': 'ds', 'Adj Close': 'y'})
prophet_df.to_csv('prophet_ready_AAPL.csv', index=False)
```

| Column | Meaning |
| :--- | :--- |
| `ds` | Date (Prophet's required name) |
| `y` | Adj Close Price (Prophet's required name) |

---

## 3️⃣ Step 3: Prophet — Train, Test, Forecast

### Why Prophet?

- Prophet understands **trend + yearly seasonality + weekly seasonality + holidays** automatically.
- Linear Regression cannot understand seasonality over time.
- SARIMA needs manual p,d,q tuning and is hard to explain in interview.
- Prophet gives `yhat` + `yhat_lower` + `yhat_upper` — forecast + confidence band.

### Train / Test Split

```python
from prophet import Prophet

train = prophet_df.iloc[:-90]  # 2014 to ~Sep 2023
test  = prophet_df.iloc[-90:]  # Last 90 days for testing

model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
model.fit(train)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)
```

### Results — Last 90 Days Test

| Metric | Value |
| :--- | :--- |
| **MAPE** | **7.25%** |
| **Accuracy** | **92.75%** (100 - MAPE) |
| **Train Rows** | 2,424 |
| **Test Rows** | 90 |

**What MAPE means in plain words:**
> On average, Prophet's prediction was only 7.25% away from the real price. For stock data, which is volatile, this is a strong result.

I also tested 200 days: MAPE 13.54% (86.46% accuracy). Longer test = harder test, so 90 days is the locked final test.

### Final Forecast Till Aug 2027

```python
# Retrain on FULL data for real future forecast
model_final = Prophet(yearly_seasonality=True, weekly_seasonality=True)
model_final.fit(prophet_df)
future_final = model_final.make_future_dataframe(periods=1300) # ~ till Aug 2027
forecast_final = model_final.predict(future_final)
forecast_final[['ds','yhat','yhat_lower','yhat_upper']].to_csv('AAPL_forecast.csv', index=False)
```

**Why 2027, not 2025?**
> In 2026, forecasting 2025 has no business value — 2025 is already past. A business wants to know the *future*. So I forecasted till Aug 2027. The confidence band widens further out — that is expected. Forecast is for direction, not exact price promise.

---

## 4️⃣ Step 4: Deployment

### Streamlit App — Interactive

- Shows Cleaned Data table
- Button: **Run Prophet Forecast**
- Charts: Actual vs Forecast, Trend, Yearly Seasonality, Weekly Seasonality
- Download button: **Download Forecast CSV for Tableau**
- Input: `prophet_ready_AAPL.csv` (ds, y)

```bash
cd aapl-streamlit
pip install -r requirements.txt
streamlit run app.py
# Open http://localhost:8501
```

### Tableau Public Dashboard — Storytelling

- **Input:** `AAPL_forecast.csv`
- **Visuals:**
  - Actual vs Forecast line chart
  - KPI Cards: MAPE 7.25%, Accuracy 92.75%, Forecast Till Aug 2027
  - Insight: *AAPL shows long-term upward trend with seasonal dips*
- **Link:** See Live Links at top

**Difference:**
> Streamlit = for technical user who wants to run model and download data.
> Tableau = for business user who wants to see story in 10 seconds.

---

## 📁 Folder Structure

```
Stock_price_prediction/
├── aapl-streamlit/
│   ├── app.py                  # Streamlit app
│   ├── requirements.txt        # streamlit, pandas, prophet, plotly, matplotlib
│   └── data/
│       ├── prophet_ready_AAPL.csv   # Input for Streamlit (ds, y)
│       └── AAPL_forecast.csv       # Output for Tableau (ds, yhat, yhat_lower, yhat_upper)
├── AAPL_clean.csv              # Full cleaned AAPL data
├── prophet_ready_AAPL.csv      # Prophet-ready file
├── AAPL_forecast.csv           # Forecast file
└── README.md
```

---

## 💡 What I Learned

1. **SQL first, Model second** — 20 KPIs in DuckDB told me AAPL is growing, volatile, mostly bullish. Without that, Prophet is a black box.
2. **Cleaning is not boring** — Sorting, deduping, using Adj Close instead of Close changes results.
3. **Feature Engineering explains the stock** — SMA, returns, intraday range are not just columns, they are business questions.
4. **Longer test = harder test** — 90 days 92.75% vs 200 days 86.46% — both correct, different test lengths.
5. **Forecast the future, not the past** — 2027 forecast has business value, 2025 forecast in 2026 does not.
6. **Two deployments, two audiences** — Streamlit for interaction, Tableau for story.

---

## 🔮 Future Scope

- Update data to 2025-2026 using `yfinance` for live forecasting
- Add ticker selector (MSFT, AMZN) in Streamlit
- Compare Prophet vs XGBoost / LSTM
- Add news sentiment as external regressor in Prophet

---

## 👤 Author

**Nayana**
- GitHub: https://github.com/nayana-n3903
- Project: AAPL Stock Price Forecasting — DuckDB SQL (20 KPIs) + Pandas (Cleaning + Feature Engineering) + Prophet + Streamlit + Tableau

---

## 🏷️ Skills Demonstrated

`DuckDB` `SQL` `Pandas` `Python` `KPI Analysis` `Feature Engineering` `Data Cleaning` `Prophet` `Time Series Forecasting` `Streamlit` `Tableau Public` `MAPE` `Data Visualization`
