import pandas as pd
import numpy as np
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_psi(training_dist: np.ndarray, current_dist: np.ndarray,
                n_bins: int = 10) -> float:
    """Returns PSI between training and current feature distributions.

    PSI < 0.10      = stable.
    PSI 0.10-0.25   = moderate drift, log a warning.
    PSI > 0.25      = significant drift, alert and consider retraining.

    Percentile-based bins ensure each training bucket has equal
    representation, preventing empty bins that would cause log(0) errors.
    """
    breakpoints     = np.percentile(training_dist, np.linspace(0, 100, n_bins + 1))
    train_counts, _ = np.histogram(training_dist, bins=breakpoints)
    curr_counts, _  = np.histogram(current_dist,  bins=breakpoints)

    train_pct = np.clip(train_counts / len(training_dist), 1e-6, None)
    curr_pct  = np.clip(curr_counts  / len(current_dist),  1e-6, None)

    return float(np.sum((curr_pct - train_pct) * np.log(curr_pct / train_pct)))


def detect_data_drift(training_features: pd.DataFrame,
                      recent_features: pd.DataFrame,
                      breach_count_limit: int) -> dict:
    """Returns data drift summary: whether drift fired and which features drifted."""
    psi_per_feature = {
        col: compute_psi(training_features[col].dropna().values,
                         recent_features[col].dropna().values)
        for col in training_features.columns
    }
    psi_series         = pd.Series(psi_per_feature)
    n_features_drifted = int((psi_series > 0.25).sum())
    drift_detected     = n_features_drifted > breach_count_limit

    return {
        "drift_detected":       drift_detected,
        "n_features_drifted":   n_features_drifted,
        "max_psi":              float(psi_series.max()),
        "top_drifted_features": psi_series.nlargest(5).to_dict(),
    }


def compute_normal_window_distances(returns: pd.DataFrame, is_normal: pd.Series,
                                    baseline_corr: np.ndarray, window: int) -> np.ndarray:
    """Returns Frobenius distances between baseline_corr and each contiguous normal window.

    Windows are walked positionally over the full, calendar-ordered `returns` table and
    skipped entirely if any day inside touches an excluded (non-normal) day. Walking a
    row-dropped, normal-only table instead would let a window silently splice trading
    days from before and after an exclusion gap (e.g. the year-long 2022 bear-market
    exclusion) into one fabricated correlation matrix — the same contiguity bug the
    sequencer guards against when building training windows.
    """
    distances = []
    for t in range(window - 1, len(returns)):
        if not is_normal.iloc[t - window + 1 : t + 1].all():
            continue
        window_corr = returns.iloc[t - window + 1 : t + 1].corr().values
        distances.append(np.linalg.norm(window_corr - baseline_corr, "fro"))
    return np.array(distances)


def detect_corr_drift(returns: pd.DataFrame, is_normal: pd.Series,
                      recent_returns: pd.DataFrame, window: int = 30) -> dict:
    """Returns whether correlation structure has shifted from the normal-day baseline.

    The baseline correlation matrix is computed from normal days only — calibrating
    against anomalous periods would treat known correlation breakdowns as part of the
    expected structure, defeating the point of this check. The threshold is calibrated
    from the same kind of distance, walked window-by-window over the full contiguous
    history via `compute_normal_window_distances`, so the calibration set isn't
    corrupted by windows spliced across an exclusion gap.
    """
    is_normal      = is_normal.reindex(returns.index, fill_value=False)
    normal_returns = returns.loc[is_normal[is_normal].index]
    baseline_corr  = normal_returns.corr().values

    current_corr  = recent_returns.corr().values
    corr_distance = float(np.linalg.norm(baseline_corr - current_corr, "fro"))

    historical_distances = compute_normal_window_distances(returns, is_normal, baseline_corr, window)
    threshold      = float(np.mean(historical_distances) + 2 * np.std(historical_distances))
    drift_detected = corr_distance > threshold

    return {
        "drift_detected": drift_detected,
        "corr_distance":  corr_distance,
        "threshold":      threshold,
    }


def detect_model_drift(recent_scores: pd.Series, baseline_scores: pd.Series,
                       anomaly_threshold: float, window_days: int = 30) -> dict:
    """Returns model drift indicators comparing recent output behavior to baseline.

    Two failure modes:
    1. Mean inflation: model outputs higher scores during normal-looking days.
    2. Flag rate spike: percentage of days flagged has grown significantly.
    Both indicate calibration drift and warrant retraining.
    """
    baseline_mean      = float(baseline_scores.mean())
    baseline_std       = float(baseline_scores.std())
    baseline_flag_rate = float((baseline_scores > anomaly_threshold).mean())

    recent_window    = recent_scores.tail(window_days)
    recent_mean      = float(recent_window.mean())
    recent_flag_rate = float((recent_window > anomaly_threshold).mean())

    mean_has_inflated    = recent_mean > baseline_mean + 2 * baseline_std
    flag_rate_has_spiked = recent_flag_rate > 3 * baseline_flag_rate

    return {
        "mean_drift_detected":      mean_has_inflated,
        "flag_rate_drift_detected": flag_rate_has_spiked,
        "recent_mean":              recent_mean,
        "baseline_mean":            baseline_mean,
        "recent_flag_rate":         recent_flag_rate,
        "baseline_flag_rate":       baseline_flag_rate,
    }


def run_drift_detection():
    config       = load_config()
    alert_config = config["alerting"]

    training_features = pd.read_csv(
        "data/processed/features_normal.csv", index_col=0, parse_dates=True
    )
    portfolio_trace = pd.read_csv(
        "data/processed/portfolio_trace.csv", index_col=0, parse_dates=True
    ).squeeze()
    returns = pd.read_csv("data/processed/returns.csv", index_col=0, parse_dates=True)
    is_normal = pd.read_csv(
        "data/processed/normal_mask.csv", index_col=0, parse_dates=True
    )["is_normal"].reindex(returns.index, fill_value=False)

    recent_features = training_features.tail(30)
    recent_returns  = returns.tail(30)

    data_drift = detect_data_drift(training_features, recent_features,
                                    alert_config["psi_feature_breach_count"])
    corr_drift = detect_corr_drift(returns, is_normal, recent_returns)

    # Calibrate model drift against normal-day scores only — the unfiltered
    # portfolio_trace history includes the excluded crisis periods (e.g. the
    # COVID spike), and a single outlier that large inflates baseline_std
    # enough to make the mean-inflation check untriggerable.
    normal_baseline_scores = portfolio_trace.loc[
        portfolio_trace.index.intersection(is_normal[is_normal].index)
    ]
    model_drift = detect_model_drift(
        portfolio_trace,
        normal_baseline_scores,
        alert_config["trace_score_threshold"],
    )

    print(f"Data drift:  detected={data_drift['drift_detected']}  "
          f"({data_drift['n_features_drifted']} features)")
    print(f"Corr drift:  detected={corr_drift['drift_detected']}  "
          f"(distance={corr_drift['corr_distance']:.3f})")
    print(f"Model drift: mean={model_drift['mean_drift_detected']}  "
          f"flag_rate={model_drift['flag_rate_drift_detected']}")


if __name__ == "__main__":
    run_drift_detection()