import os

from yahoo_finance_api.stock_universe import download_stock_list_to_csv


if __name__ == "__main__":
    source = os.getenv("STOCK_META_SOURCE", "yahoo")
    fmp_key = os.getenv("FMP_API_KEY")
    df = download_stock_list_to_csv(
        output_csv="latest_us_stock_list.csv",
        metadata_source=source,
        fmp_api_key=fmp_key,
    )
    print(f"Saved {len(df)} rows to latest_us_stock_list.csv using source={source}")
