from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from . import YahooFinance


@dataclass
class ModelArtifacts:
    model: GradientBoostingRegressor
    feature_columns: list[str]
    train_rows: int
    test_rows: int
    mae: float
    r2: float


class StockForecastModel:
    """Build a one-month stock price-change forecasting model.

    Features include moving averages, price-weighted volume aggregates,
    one-week and one-month momentum, and external market features from
    gold and VIX weekly changes.
    """

    def __init__(
        self,
        stock_ticker: str,
        market_cap_usd: float,
        category_label: str,
        history_range: str = "5y",
    ) -> None:
        self.stock_ticker = stock_ticker
        self.market_cap_usd = market_cap_usd
        self.category_label = category_label
        self.history_range = history_range
        self.data: Optional[pd.DataFrame] = None
        self.artifacts: Optional[ModelArtifacts] = None
        self._eval_cache: Optional[pd.DataFrame] = None

    def load_raw_data(self) -> pd.DataFrame:
        stock = YahooFinance(self.stock_ticker, result_range=self.history_range, interval="1d").result
        gold = YahooFinance("GC=F", result_range=self.history_range, interval="1d").result
        vix = YahooFinance("^VIX", result_range=self.history_range, interval="1d").result

        df = stock[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["gold_close"] = gold["Close"].reindex(df.index).ffill()
        df["vix_close"] = vix["Close"].reindex(df.index).ffill()
        self.data = df.dropna().copy()
        return self.data

    def build_features(self) -> pd.DataFrame:
        if self.data is None:
            self.load_raw_data()

        df = self.data.copy()
        df["ma_20"] = df["Close"].rolling(20).mean()
        df["ma_120"] = df["Close"].rolling(120).mean()
        df["ma_200"] = df["Close"].rolling(200).mean()

        df["price_weighted_volume"] = df["Close"] * df["Volume"]
        df["pwv_1w"] = df["price_weighted_volume"].rolling(5).mean()
        df["pwv_1m"] = df["price_weighted_volume"].rolling(21).mean()

        df["change_1w"] = df["Close"].pct_change(5)
        df["change_1m"] = df["Close"].pct_change(21)
        df["gold_change_1w"] = df["gold_close"].pct_change(5)
        df["vix_change_1w"] = df["vix_close"].pct_change(5)

        df["stock_category"] = self.category_label
        df["is_large_cap_10b"] = float(self.market_cap_usd > 10_000_000_000)

        df["target_1m_price_change"] = df["Close"].shift(-21) / df["Close"] - 1
        df = pd.get_dummies(df, columns=["stock_category"], drop_first=False)

        feature_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "ma_20",
            "ma_120",
            "ma_200",
            "pwv_1w",
            "pwv_1m",
            "change_1w",
            "change_1m",
            "gold_change_1w",
            "vix_change_1w",
            "is_large_cap_10b",
        ] + [c for c in df.columns if c.startswith("stock_category_")]

        engineered = df[feature_cols + ["target_1m_price_change"]].dropna()
        self.data = engineered
        return engineered


    def plot_historical_trend(self, window_size: int = 180):
        """Plot historical close trend with MA5/MA20/MA120.

        :param window_size: Number of most recent rows to plot.
        :return: matplotlib Axes object
        """
        if self.data is None:
            self.load_raw_data()

        base = self.data[["Close"]].copy()
        base["MA5"] = base["Close"].rolling(5).mean()
        base["MA20"] = base["Close"].rolling(20).mean()
        base["MA120"] = base["Close"].rolling(120).mean()
        plot_df = base.tail(window_size).dropna()

        ax = plot_df[["Close", "MA5", "MA20", "MA120"]].plot(
            figsize=(12, 6),
            title=f"{self.stock_ticker} Historical Trend (last {window_size} periods)"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(alpha=0.3)
        return ax

    def train(self, random_state: int = 42) -> ModelArtifacts:
        if self.data is None or "target_1m_price_change" not in self.data.columns:
            self.build_features()

        df = self.data.copy()
        X = df.drop(columns=["target_1m_price_change"])
        y = df["target_1m_price_change"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            shuffle=False,
        )

        model = GradientBoostingRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        artifacts = ModelArtifacts(
            model=model,
            feature_columns=list(X.columns),
            train_rows=len(X_train),
            test_rows=len(X_test),
            mae=float(mean_absolute_error(y_test, preds)),
            r2=float(r2_score(y_test, preds)),
        )
        self.artifacts = artifacts
        return artifacts


    def evaluate_performance(self, test_size: float = 0.2, random_state: int = 42) -> dict:
        """Evaluate model with MAE and RMSE and cache prediction frame."""
        if self.data is None or "target_1m_price_change" not in self.data.columns:
            self.build_features()

        df = self.data.copy()
        X = df.drop(columns=["target_1m_price_change"])
        y = df["target_1m_price_change"]

        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = GradientBoostingRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        eval_df = pd.DataFrame({
            "actual": y_test.values,
            "predicted": preds,
        }, index=y_test.index)
        eval_df["error"] = eval_df["predicted"] - eval_df["actual"]
        self._eval_cache = eval_df

        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(mean_squared_error(y_test, preds) ** 0.5)
        return {"mae": mae, "rmse": rmse, "test_rows": len(y_test)}

    def plot_performance(self, test_size: float = 0.2, random_state: int = 42):
        """Plot actual vs predicted returns on the test window."""
        if self._eval_cache is None:
            self.evaluate_performance(test_size=test_size, random_state=random_state)

        ax = self._eval_cache[["actual", "predicted"]].plot(
            figsize=(12, 6),
            title=f"{self.stock_ticker} Test Set: Actual vs Predicted 1-Month Return"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("1-Month Return")
        ax.grid(alpha=0.3)
        return ax

    def plot_prediction_scatter(self, test_size: float = 0.2, random_state: int = 42):
        """Scatter plot of actual vs predicted with y=x reference line."""
        if self._eval_cache is None:
            self.evaluate_performance(test_size=test_size, random_state=random_state)

        ax = self._eval_cache.plot.scatter(x="actual", y="predicted", figsize=(7, 7), alpha=0.6)
        lim_low = min(self._eval_cache["actual"].min(), self._eval_cache["predicted"].min())
        lim_high = max(self._eval_cache["actual"].max(), self._eval_cache["predicted"].max())
        ax.plot([lim_low, lim_high], [lim_low, lim_high], "r--", linewidth=1)
        ax.set_title(f"{self.stock_ticker} Actual vs Predicted Scatter")
        ax.grid(alpha=0.3)
        return ax

    def backtest_predictions(self, min_train_size: int = 252, step: int = 1) -> pd.DataFrame:
        """Walk-forward backtest: predict next 1-month return for each day.

        Uses only historical data available up to each forecast date, then compares
        with the realized 1-month forward return.
        """
        if self.data is None or "target_1m_price_change" not in self.data.columns:
            self.build_features()

        df = self.data.copy()
        X_all = df.drop(columns=["target_1m_price_change"])
        y_all = df["target_1m_price_change"]

        records = []
        model = GradientBoostingRegressor(random_state=42)

        for i in range(min_train_size, len(df), step):
            X_train = X_all.iloc[:i]
            y_train = y_all.iloc[:i]
            X_test = X_all.iloc[[i]]

            model.fit(X_train, y_train)
            pred = float(model.predict(X_test)[0])
            actual = float(y_all.iloc[i])

            records.append({
                "date": df.index[i],
                "predicted_1m_change": pred,
                "actual_1m_change": actual,
            })

        backtest_df = pd.DataFrame(records).set_index("date")
        backtest_df["prediction_error"] = backtest_df["predicted_1m_change"] - backtest_df["actual_1m_change"]
        return backtest_df

    def plot_backtest(self, min_train_size: int = 252, step: int = 1):
        """Plot historical backtest with predicted vs actual 1-month return."""
        backtest_df = self.backtest_predictions(min_train_size=min_train_size, step=step)
        ax = backtest_df[["actual_1m_change", "predicted_1m_change"]].plot(
            figsize=(12, 6),
            title=f"{self.stock_ticker} Backtest: Actual vs Predicted 1-Month Change"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("1-Month Return")
        ax.grid(alpha=0.3)
        return ax

    def predict_latest(self) -> float:
        if self.artifacts is None:
            raise RuntimeError("Call train() before predict_latest().")

        latest = self.data[self.artifacts.feature_columns].tail(1)
        return float(self.artifacts.model.predict(latest)[0])


def build_multi_stock_dataset(configs: Iterable[dict], history_range: str = "5y") -> pd.DataFrame:
    """Create a combined dataset across several stocks in the same category schema.

    Each config should include: ticker, market_cap_usd, category_label.
    """

    frames: list[pd.DataFrame] = []
    for cfg in configs:
        model = StockForecastModel(
            stock_ticker=cfg["ticker"],
            market_cap_usd=float(cfg["market_cap_usd"]),
            category_label=cfg["category_label"],
            history_range=history_range,
        )
        frames.append(model.build_features())

    return pd.concat(frames, axis=0).dropna()
