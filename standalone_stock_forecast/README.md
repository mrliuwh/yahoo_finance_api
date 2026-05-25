# Standalone Stock Forecast Model

This folder is structured so you can copy it into a **new GitHub repository** and run it independently.

## What it does
- Downloads daily OHLCV stock data from Yahoo Finance.
- Builds engineered features:
  - 20/120/200-day moving averages
  - price-weighted volume + 1-week / 1-month rolling averages
  - 1-week / 1-month momentum
  - weekly gold change
  - weekly VIX change
  - stock category (one-hot encoded)
  - market cap bucket (>10B)
- Trains a **GradientBoostingRegressor** to predict the next 1-month (21 trading days) return.
- Includes historical trend plotting with MA5, MA20, and MA120 lines.

## Project structure
- `yahoo_client.py` - lightweight Yahoo chart API downloader
- `forecast_model.py` - feature engineering + training/inference code
- `train_example.py` - runnable example
- `requirements.txt` - Python dependencies
- `stock_list_downloader.py` - download latest stock list + market cap + category to CSV

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_example.py
```

## Move to a new repo
1. Create a new empty GitHub repo.
2. Copy all files from `standalone_stock_forecast/` into that repo root.
3. Commit and push.


## Plot historical trend
```python
from forecast_model import StockForecastModel

model = StockForecastModel("AAPL", 2_900_000_000_000, "technology")
model.load_raw_data()
ax = model.plot_historical_trend(window_size=250)
```


## Backtest historical forecasts
```python
bt = model.backtest_predictions(min_train_size=252, step=1)
ax = model.plot_backtest(min_train_size=252, step=1)
```


## Download latest stock list CSV
```bash
python stock_list_downloader.py
```


Use different metadata providers:

```python
# Yahoo (default)
df = download_stock_list_to_csv("latest_us_stock_list.csv", metadata_source="yahoo")

# Financial Modeling Prep (recommended fallback)
df = download_stock_list_to_csv("latest_us_stock_list.csv", metadata_source="fmp", fmp_api_key="YOUR_KEY")
```
