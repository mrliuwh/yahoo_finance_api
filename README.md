# Update - I am not maintaining this project so if you have any problem please fork this project and try solving it yourselves.

# Yahoo Finance Python API

Due to rampant deprecation of all the stocks api like pandas_datareader etc, i have to created my own based on the yahoo api v8

The best thing about this api is that it support **Intraday Data** upto 1 minute granularity which a lots of free api doesn't support

## Usage

This command returns a dataframe which can be further modified to add new columns , exported to excel etc

``` python
tata_power = YahooFinance('TATAPOWER.NS', result_range='1mo', interval='15m', dropna='True').result
```


### Demo
![Alt Text](/res/demo.gif)

## Installation

``` bash
git clone https://github.com/mayankwadhwa/yahoo_finance_api.git
cd yahoo_finance_api
python setup.py install
```

### Requirements

- Pandas
- Request

### Note

Make sure to use TICKER symbol from yahoo finance website
https://in.finance.yahoo.com/ 

## Forecast model example (Gradient Boosting)

```python
from yahoo_finance_api import StockForecastModel

model = StockForecastModel(
    stock_ticker="AAPL",
    market_cap_usd=2_900_000_000_000,
    category_label="technology"
)

artifacts = model.train()
print("MAE:", artifacts.mae)
print("R2:", artifacts.r2)
print("Predicted next 1-month change:", model.predict_latest())
```

The model uses daily OHLCV, 20/120/200-day moving averages, price-weighted
volume for 1-week and 1-month windows, weekly gold and VIX changes, and a
large-cap (>10B) indicator with stock category encoding.


You can also visualize historical trend with MA5, MA20, MA120:

```python
model.load_raw_data()
ax = model.plot_historical_trend(window_size=250)
```


Backtest (walk-forward, predicted vs actual 1-month return):

```python
bt = model.backtest_predictions(min_train_size=252, step=1)
ax = model.plot_backtest(min_train_size=252, step=1)
```


## Download latest stock list to CSV

```python
from yahoo_finance_api import download_stock_list_to_csv

df = download_stock_list_to_csv("latest_us_stock_list.csv")
print(df.head())
```

This CSV includes stock symbol list, exchange, market cap (market size), and stock category (sector).


Use different metadata providers:

```python
# Yahoo (default)
df = download_stock_list_to_csv("latest_us_stock_list.csv", metadata_source="yahoo")

# Financial Modeling Prep (recommended fallback)
df = download_stock_list_to_csv("latest_us_stock_list.csv", metadata_source="fmp", fmp_api_key="YOUR_KEY")
```


Model evaluation (MAE/RMSE) and prediction plots:

```python
metrics = model.evaluate_performance(test_size=0.2)
print(metrics)  # {"mae": ..., "rmse": ..., "test_rows": ...}

ax1 = model.plot_performance(test_size=0.2)
ax2 = model.plot_prediction_scatter(test_size=0.2)
```


Feature importance check:

```python
fi = model.get_feature_importance(top_n=20)
print(fi.head(10))
ax = model.plot_feature_importance(top_n=20)
```
