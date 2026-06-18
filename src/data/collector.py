import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_end_date(end_date: str) -> str:
    """Returns the date to pass as yfinance's `end` parameter.

    yfinance treats `end` as exclusive, so to include today's bar we
    pass tomorrow. Running Friday afternoon with end=Friday returns
    data only through Thursday — TRACE would silently lag the market
    by one day.
    """
    if end_date == "today":
        tomorrow = pd.Timestamp.today() + pd.Timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")
    return end_date


def download_ohlcv(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Returns raw OHLCV DataFrame for all tickers from Yahoo Finance.

    threads=False avoids a yfinance/Windows race where concurrent tickers
    writing to its shared tz-cache sqlite db intermittently raises
    'database is locked', silently dropping one ticker's data.
    """
    return yf.download(tickers, start=start, end=end, auto_adjust=True, threads=False)


def save_close_prices(raw: pd.DataFrame, output_dir: Path) -> None:
    """Writes one CSV per ticker so individual assets can be refetched."""
    output_dir.mkdir(parents=True, exist_ok=True)
    close = raw["Close"]
    for ticker in close.columns:
        if close[ticker].isna().all():
            raise ValueError(
                f"Download for ticker '{ticker}' returned no data — refusing to "
                f"write an empty raw file. Re-run the collector."
            )
        close[[ticker]].to_csv(output_dir / f"{ticker}.csv")


def collect():
    config     = load_config()
    end_date   = resolve_end_date(config["data"]["end_date"])
    tickers    = config["data"]["tickers"]
    start_date = config["data"]["start_date"]

    print(f"Downloading {len(tickers)} tickers: {start_date} -> {end_date}")
    raw = download_ohlcv(tickers, start_date, end_date)
    save_close_prices(raw, Path("data/raw"))
    print("Saved to data/raw/")


if __name__ == "__main__":
    collect()