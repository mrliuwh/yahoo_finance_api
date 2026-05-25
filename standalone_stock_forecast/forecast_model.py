from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from yahoo_client import YahooFinanceClient


@dataclass
class ModelArtifacts:
    model: GradientBoostingRegressor
    feature_columns: list[str]
    mae: float
    r2: float


class StockForecastModel:
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

    def load_raw_data(self) -> pd.DataFrame:
        stock = YahooFinanceClient(self.stock_ticker, self.history_range, "1d").fetch()
        gold = YahooFinanceClient("GC=F", self.history_range, "1d").fetch()
        vix = YahooFinanceClient("^VIX", self.history_range, "1d").fetch()

        df = stock[["Open", "High", "Low", "Close", "Volume"]].copy()
        df["gold_close"] = gold["Close"].reindex(df.index).ffill()
        df["vix_close"] = vix["Close"].reindex(df.index).ffill()
        self.data = df.dropna()
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
            "Open", "High", "Low", "Close", "Volume", "ma_20", "ma_120", "ma_200",
            "pwv_1w", "pwv_1m", "change_1w", "change_1m", "gold_change_1w", "vix_change_1w",
            "is_large_cap_10b",
        ] + [c for c in df.columns if c.startswith("stock_category_")]

        self.data = df[feature_cols + ["target_1m_price_change"]].dropna()
        return self.data


    def plot_historical_trend(self, window_size: int = 180):
        """Plot historical close trend with MA5/MA20/MA120."""
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

        X = self.data.drop(columns=["target_1m_price_change"])
        y = self.data["target_1m_price_change"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = GradientBoostingRegressor(random_state=random_state)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        self.artifacts = ModelArtifacts(
            model=model,
            feature_columns=list(X.columns),
            mae=float(mean_absolute_error(y_test, preds)),
            r2=float(r2_score(y_test, preds)),
        )
        return self.artifacts

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
            raise RuntimeError("Train model first.")
        latest = self.data[self.artifacts.feature_columns].tail(1)
        return float(self.artifacts.model.predict(latest)[0])
