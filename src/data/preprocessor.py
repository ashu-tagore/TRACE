import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_aligned_prices(raw_dir: Path, tickers: list) -> pd.DataFrame:
    """Returns prices for all tickers aligned on shared trading days.

    Inner join only. Forward-filling would inject fake data into the
    cross-asset correlation signal that TRACE is designed to detect.
    """
    frames = {}
    for ticker in tickers:
        df = pd.read_csv(raw_dir / f"{ticker}.csv", index_col=0, parse_dates=True)
        frames[ticker] = df.iloc[:, 0]
    return pd.DataFrame(frames).dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Returns log returns. First row is NaN by definition and is dropped."""
    return np.log(prices / prices.shift(1)).dropna()


def preprocess():
    config  = load_config()
    tickers = config["data"]["tickers"]

    prices  = load_aligned_prices(Path("data/raw"), tickers)
    returns = compute_log_returns(prices)

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    returns.to_csv(output_dir / "returns.csv")

    print(f"Saved returns.csv: {returns.shape[0]} days x {returns.shape[1]} assets")


if __name__ == "__main__":
    preprocess()