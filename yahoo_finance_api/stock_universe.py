from __future__ import annotations

import os
import time
from typing import Iterable

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
FMP_PROFILE_URL = "https://financialmodelingprep.com/api/v3/profile"


def _read_symbol_file(url: str, symbol_col: str) -> pd.DataFrame:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    header = lines[0].split("|")
    rows = [line.split("|") for line in lines[1:] if not line.startswith("File Creation Time")]
    return pd.DataFrame(rows, columns=header)


def fetch_us_stock_symbols() -> pd.DataFrame:
    """Fetch a broad US stock symbol list from NASDAQ Trader directories."""
    nasdaq = _read_symbol_file(NASDAQ_LISTED_URL, "Symbol")
    nasdaq = nasdaq.rename(columns={"Symbol": "symbol", "Security Name": "security_name"})
    nasdaq["exchange"] = "NASDAQ"

    other = _read_symbol_file(OTHER_LISTED_URL, "ACT Symbol")
    other = other.rename(columns={"ACT Symbol": "symbol", "Security Name": "security_name", "Exchange": "exchange"})

    combined = pd.concat([
        nasdaq[["symbol", "security_name", "exchange"]],
        other[["symbol", "security_name", "exchange"]],
    ], ignore_index=True)
    combined = combined[~combined["symbol"].str.contains(r"\$", regex=True, na=False)]
    return combined.drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def _chunks(items: Iterable[str], size: int):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _safe_fetch_yahoo_batch(batch: list[str], headers: dict, retries: int = 3) -> list[dict]:
    params = {"symbols": ",".join(batch)}
    delay = 1.0
    for attempt in range(retries):
        try:
            response = requests.get(YAHOO_QUOTE_URL, params=params, headers=headers, timeout=30)
            if response.status_code == 401:
                return []
            response.raise_for_status()
            return response.json().get("quoteResponse", {}).get("result", [])
        except requests.RequestException:
            if attempt == retries - 1:
                return []
            time.sleep(delay)
            delay *= 2
    return []


def enrich_with_yahoo_metadata(symbols_df: pd.DataFrame, batch_size: int = 200) -> pd.DataFrame:
    records = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for batch in _chunks(symbols_df["symbol"].tolist(), batch_size):
        result = _safe_fetch_yahoo_batch(batch, headers=headers)
        if not result and len(batch) > 1:
            for sym in batch:
                one = _safe_fetch_yahoo_batch([sym], headers=headers)
                if one:
                    result.extend(one)
                time.sleep(0.05)
        for item in result:
            records.append({
                "symbol": item.get("symbol"),
                "market_cap": item.get("marketCap"),
                "sector": item.get("sector"),
                "industry": item.get("industry"),
                "quote_type": item.get("quoteType"),
                "meta_source": "yahoo",
            })

    meta = pd.DataFrame(records).drop_duplicates(subset=["symbol"]) if records else pd.DataFrame(columns=["symbol", "market_cap", "sector", "industry", "quote_type", "meta_source"])
    merged = symbols_df.merge(meta, on="symbol", how="left")
    merged["stock_category"] = merged["sector"].fillna("Unknown")
    return merged


def enrich_with_fmp_metadata(symbols_df: pd.DataFrame, api_key: str | None = None, batch_size: int = 100) -> pd.DataFrame:
    """Add market cap and category from Financial Modeling Prep profiles API."""
    resolved_key = api_key or os.getenv("FMP_API_KEY")
    if not resolved_key:
        raise ValueError("FMP API key is required. Pass api_key or set FMP_API_KEY env var.")

    records: list[dict] = []
    for batch in _chunks(symbols_df["symbol"].tolist(), batch_size):
        symbols = ",".join(batch)
        url = f"{FMP_PROFILE_URL}/{symbols}"
        try:
            response = requests.get(url, params={"apikey": resolved_key}, timeout=30)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, list):
                result = []
        except requests.RequestException:
            result = []

        for item in result:
            records.append({
                "symbol": item.get("symbol"),
                "market_cap": item.get("mktCap"),
                "sector": item.get("sector"),
                "industry": item.get("industry"),
                "quote_type": item.get("ipoDate"),
                "meta_source": "fmp",
            })
        time.sleep(0.1)

    meta = pd.DataFrame(records).drop_duplicates(subset=["symbol"]) if records else pd.DataFrame(columns=["symbol", "market_cap", "sector", "industry", "quote_type", "meta_source"])
    merged = symbols_df.merge(meta, on="symbol", how="left")
    merged["stock_category"] = merged["sector"].fillna("Unknown")
    return merged


def download_stock_list_to_csv(
    output_csv: str = "latest_us_stock_list.csv",
    metadata_source: str = "yahoo",
    fmp_api_key: str | None = None,
) -> pd.DataFrame:
    """Download latest stock list + market size + stock category and save to CSV.

    metadata_source: "yahoo" or "fmp".
    """
    symbols = fetch_us_stock_symbols()
    source = metadata_source.lower().strip()
    if source == "yahoo":
        enriched = enrich_with_yahoo_metadata(symbols)
    elif source == "fmp":
        enriched = enrich_with_fmp_metadata(symbols, api_key=fmp_api_key)
    else:
        raise ValueError("Unsupported metadata_source. Use 'yahoo' or 'fmp'.")

    enriched.to_csv(output_csv, index=False)
    return enriched
