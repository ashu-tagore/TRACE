import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_manual_exclusion_mask(index: pd.DatetimeIndex,
                                exclude_periods: list) -> pd.Series:
    """Returns True for days inside a manually-defined crisis window."""
    is_excluded = pd.Series(False, index=index)
    for period in exclude_periods:
        in_period = (index >= period["start"]) & (index <= period["end"])
        is_excluded |= in_period
    return is_excluded


def build_vol_stress_mask(returns: pd.DataFrame, sigma: float) -> pd.Series:
    """Returns True for days where portfolio vol exceeds mu + sigma * std."""
    portfolio_return = returns.mean(axis=1)
    rolling_vol      = portfolio_return.rolling(20).std()
    vol_threshold    = rolling_vol.mean() + sigma * rolling_vol.std()
    return (rolling_vol > vol_threshold).fillna(False)


def compute_daily_corr_distances(returns: pd.DataFrame,
                                 history_window: int,
                                 current_window: int) -> pd.Series:
    """Returns Frobenius distance between current and historical correlation per day.

    Both windows are inclusive of day t — pandas .iloc[a:b] is exclusive
    on the right, so we slice [t - window + 1 : t + 1] to include t
    itself. Excluding t would mean today's correlation breakdown only
    registers tomorrow.
    """
    distances = []
    dates     = []
    for t in range(history_window - 1, len(returns)):
        historical_corr = returns.iloc[t - history_window + 1 : t + 1].corr().values
        current_corr    = returns.iloc[t - current_window + 1 : t + 1].corr().values
        distance        = np.linalg.norm(historical_corr - current_corr, "fro")
        distances.append(distance)
        dates.append(returns.index[t])
    return pd.Series(distances, index=dates)


def build_corr_stress_mask(returns: pd.DataFrame,
                           history_window: int,
                           current_window: int,
                           sigma: float) -> pd.Series:
    """Returns True for days where correlation structure has broken down."""
    distances   = compute_daily_corr_distances(returns, history_window, current_window)
    threshold   = distances.mean() + sigma * distances.std()
    is_stressed = distances > threshold
    return is_stressed.reindex(returns.index, fill_value=False)


def label_normal_days(returns: pd.DataFrame, config: dict) -> pd.Series:
    """Returns boolean Series: True = normal trading day, safe to train on."""
    tc = config["training"]

    is_manually_excluded = build_manual_exclusion_mask(returns.index, tc["exclude_periods"])
    is_vol_stressed      = build_vol_stress_mask(returns, tc["vol_threshold_sigma"])
    is_corr_stressed     = build_corr_stress_mask(
        returns,
        history_window=tc["corr_stability_window"],
        current_window=tc["corr_current_window"],
        sigma=tc["corr_drift_threshold_sigma"],
    )

    is_anomalous = is_manually_excluded | is_vol_stressed | is_corr_stressed
    return ~is_anomalous


def filter_normal_data():
    config   = load_config()
    returns  = pd.read_csv("data/processed/returns.csv",  index_col=0, parse_dates=True)
    features = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)

    shared_index = returns.index.intersection(features.index)
    returns      = returns.loc[shared_index]
    features     = features.loc[shared_index]

    is_normal = label_normal_days(returns, config)

    output_dir = Path("data/processed")
    features[is_normal].to_csv(output_dir / "features_normal.csv")
    is_normal.to_csv(output_dir / "normal_mask.csv", header=["is_normal"])

    normal_pct = is_normal.mean() * 100
    print(f"Normal:  {is_normal.sum()} days ({normal_pct:.1f}%)")
    print(f"Anomaly: {(~is_normal).sum()} days ({100 - normal_pct:.1f}%)")


if __name__ == "__main__":
    filter_normal_data()