"""
forecast.py
Forecasts next 3 months of total spending using Prophet, based on
historical monthly spending totals.
"""

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data/synthetic/transactions.csv')
df['date'] = pd.to_datetime(df['date'])

# Only forecast spending, not income
spending = df[df['category'] != 'Income'].copy()

# Aggregate total spending by month
monthly = spending.groupby(spending['date'].dt.to_period('M'))['amount'].sum().reset_index()
monthly['date'] = monthly['date'].dt.to_timestamp()
monthly.columns = ['ds', 'y']

print("Monthly spending totals:")
print(monthly)
print(f"\nNumber of months of data: {len(monthly)}")

# NOTE: yearly_seasonality is disabled because we only have 18 months of
# history — not enough to reliably learn a true yearly cycle. Enabling it
# caused the model to overfit noise as "seasonality," producing unstable
# month-to-month swings (including nonsensical negative spending).
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    growth='flat'
)
model.fit(monthly)

future = model.make_future_dataframe(periods=3, freq='MS')
forecast = model.predict(future)
forecast['yhat'] = forecast['yhat'].clip(lower=0)
forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)

print("\nForecast (last 3 actual months + 3 future months):")
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(6))

os.makedirs("reports", exist_ok=True)
fig = model.plot(forecast)
plt.title("Monthly Spending Forecast")
plt.xlabel("Month")
plt.ylabel("Total Spending ($)")
plt.tight_layout()
plt.savefig("reports/spending_forecast.png", dpi=150)
print("\nSaved forecast plot to reports/spending_forecast.png")