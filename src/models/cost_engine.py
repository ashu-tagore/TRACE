import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import mlflow.pytorch
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_production_model():
    """Returns the model version currently holding the 'production' alias in MLflow."""
    return mlflow.pytorch.load_model("models:/TRACE-LSTMVAE@production")


def compute_reconstruction_errors(model, features_normalized: np.ndarray,
                                  seq_len: int, n_assets: int,
                                  batch_size: int = 128) -> np.ndarray:
    """Returns per-asset MSE at the last timestep for each window.

    Shape: [n_days - seq_len + 1, n_assets].

    Only the first n_assets columns (log returns) are reconstruction
    targets. Vol and correlation features are inputs that provide context.

    We evaluate only the last timestep of each window — "today" — because
    we care whether the model can reconstruct today given its preceding
    context, not the average reconstruction quality across the window.
    """
    model.eval()
    device = next(model.parameters()).device

    n_windows = len(features_normalized) - seq_len + 1
    windows   = np.stack([
        features_normalized[start : start + seq_len]
        for start in range(n_windows)
    ]).astype(np.float32)
    windows_tensor = torch.from_numpy(windows)

    loader = DataLoader(
        TensorDataset(windows_tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    batched_errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch       = batch.to(device)
            recon, _, _ = model(batch)
            last_recon  = recon[:, -1, :n_assets]
            last_true   = batch[:, -1, :n_assets]
            mse_per_asset = ((last_true - last_recon) ** 2).cpu().numpy()
            batched_errors.append(mse_per_asset)

    return np.concatenate(batched_errors, axis=0)


def compute_vol_regime_weights(returns: pd.DataFrame, window: int) -> pd.Series:
    """Returns weight > 1 during calm regimes and < 1 during stressed regimes."""
    portfolio_return = returns.mean(axis=1)
    rolling_vol      = portfolio_return.rolling(window).std()
    historical_avg   = rolling_vol.mean()

    # guard against zero rolling vol producing inf weights — flat windows
    # (synthetic data, holiday-adjacent gaps) fall back to the historical average
    safe_denominator = rolling_vol.fillna(historical_avg).replace(0, historical_avg)
    return historical_avg / safe_denominator


def compute_correlation_penalties(returns: pd.DataFrame,
                                  history_window: int,
                                  current_window: int) -> pd.Series:
    """Returns daily Frobenius distance between current and historical correlation.

    Both windows include day t. Pandas `.iloc[a:b]` is exclusive on b,
    so we slice `[t - window + 1 : t + 1]` to actually include today.
    Without the +1, today's correlation breakdown wouldn't show up in
    the penalty until tomorrow — defeating the whole point of a
    same-day risk score.
    """
    penalties = []
    dates     = []
    for t in range(history_window - 1, len(returns)):
        historical_corr = returns.iloc[t - history_window + 1 : t + 1].corr().values
        current_corr    = returns.iloc[t - current_window + 1 : t + 1].corr().values
        penalties.append(np.linalg.norm(historical_corr - current_corr, "fro"))
        dates.append(returns.index[t])
    return pd.Series(penalties, index=dates)


def compute_persistence_weights(base_scores: pd.DataFrame, decay: float) -> pd.DataFrame:
    """Returns EMA of past anomaly scores — persistent anomalies cost more than spikes."""
    return base_scores.ewm(alpha=1 - decay).mean()


def compute_trace_scores(base_errors: pd.DataFrame,
                         returns: pd.DataFrame,
                         ce_config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Returns (per_asset_trace_scores, portfolio_trace_score).

    Requires returns to span at least corr_rolling_window days; shorter
    inputs produce an empty result because the correlation penalty needs
    that much history to compute.
    """
    vol_weight   = compute_vol_regime_weights(returns, ce_config["vol_rolling_window"])
    corr_penalty = compute_correlation_penalties(
        returns,
        history_window=ce_config["corr_rolling_window"],
        current_window=ce_config["corr_current_window"],
    )
    persistence  = compute_persistence_weights(base_errors, ce_config["persistence_decay"])

    shared_index = (base_errors.index
                    .intersection(vol_weight.index)
                    .intersection(corr_penalty.index))

    base    = base_errors.loc[shared_index]
    vol_w   = vol_weight.loc[shared_index]
    corr_p  = corr_penalty.loc[shared_index]
    persist = persistence.loc[shared_index]

    n_assets        = base.shape[1]
    exposure_weight = 1.0 / n_assets

    trace_scores = (
        base
        .multiply(vol_w,        axis=0)
        .multiply(1 + corr_p,   axis=0)
        .multiply(persist)
        * exposure_weight
    )

    portfolio_trace = trace_scores.sum(axis=1)
    return trace_scores, portfolio_trace


def run_cost_engine():
    config  = load_config()
    seq_len = config["data"]["seq_len"]
    tickers = config["data"]["tickers"]

    features = pd.read_csv("data/processed/features.csv", index_col=0, parse_dates=True)
    returns  = pd.read_csv("data/processed/returns.csv",  index_col=0, parse_dates=True)
    mean     = pd.read_csv("data/processed/feature_mean.csv", index_col=0).squeeze()
    std      = pd.read_csv("data/processed/feature_std.csv",  index_col=0).squeeze()

    features_normalized = ((features - mean) / std).values

    model  = load_production_model()
    errors = compute_reconstruction_errors(model, features_normalized, seq_len, len(tickers))

    error_index = features.index[seq_len - 1:]
    base_errors = pd.DataFrame(errors, index=error_index, columns=tickers)

    trace_scores, portfolio_trace = compute_trace_scores(
        base_errors, returns, config["cost_engine"]
    )

    output_dir = Path("data/processed")
    trace_scores.to_csv(output_dir    / "trace_scores.csv")
    portfolio_trace.to_csv(output_dir / "portfolio_trace.csv", header=["trace_score"])

    print(f"Max portfolio TRACE: {portfolio_trace.max():.4f} on {portfolio_trace.idxmax()}")


if __name__ == "__main__":
    run_cost_engine()