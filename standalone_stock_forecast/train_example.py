from forecast_model import StockForecastModel


if __name__ == "__main__":
    model = StockForecastModel(
        stock_ticker="AAPL",
        market_cap_usd=2_900_000_000_000,
        category_label="technology",
        history_range="5y",
    )

    artifacts = model.train()
    prediction = model.predict_latest()

    print("Model trained successfully")
    print("MAE:", artifacts.mae)
    print("R2:", artifacts.r2)
    print("Predicted 1-month forward return:", prediction)
