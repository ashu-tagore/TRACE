import pandas as pd
import yaml
from pathlib import Path
from itertools import combinations


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_rolling_volatility(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Returns rolling std of log returns per asset. Shape: [n_days, n_assets]."""
    vol = returns.rolling(window).std()
    vol.columns = [f"{col}_vol_{window}d" for col in vol.columns]
    return vol


def compute_rolling_pairwise_correlations(returns: pd.DataFrame,
                                          window: int) -> pd.DataFrame:
    """Returns rolling correlation for every asset pair. Shape: [n_days, n_pairs]."""
    pairs = list(combinations(returns.columns, 2))
    corr_series = {}
    for asset_a, asset_b in pairs:
        col_name = f"{asset_a}_{asset_b}_corr_{window}d"
        corr_series[col_name] = returns[asset_a].rolling(window).corr(returns[asset_b])
    return pd.DataFrame(corr_series)


def engineer():
    config      = load_config()
    corr_window = config["training"]["corr_stability_window"]

    returns = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)

    vol_5d  = compute_rolling_volatility(returns, window=5)
    vol_20d = compute_rolling_volatility(returns, window=20)
    corr    = compute_rolling_pairwise_correlations(returns, window=corr_window)

    features = pd.concat([returns, vol_5d, vol_20d, corr], axis=1).dropna()

    output_dir = Path("data/processed")
    features.to_csv(output_dir / "features.csv")
    print(f"Saved features.csv: {features.shape[0]} days x {features.shape[1]} features")


if __name__ == "__main__":
    engineer()