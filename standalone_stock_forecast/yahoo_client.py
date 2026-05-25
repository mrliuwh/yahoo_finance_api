from __future__ import annotations

import time as _time

import pandas as pd
import requests


class YahooFinanceClient:
    def __init__(self, ticker: str, result_range: str = "1y", interval: str = "1d") -> None:
        self.ticker = ticker
        self.result_range = result_range
        self.interval = interval

    def fetch(self) -> pd.DataFrame:
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {"range": self.result_range, "interval": self.interval}
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}"

        response = requests.get(url=url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        error = data["chart"]["error"]
        if error:
            raise ValueError(error["description"])

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]

        index = [_time.strftime("%Y-%m-%d", _time.localtime(ts)) for ts in timestamps]
        df = pd.DataFrame(
            {
                "Open": quote["open"],
                "High": quote["high"],
                "Low": quote["low"],
                "Close": quote["close"],
                "Volume": quote["volume"],
            },
            index=pd.to_datetime(index),
        )

        return df.dropna()
