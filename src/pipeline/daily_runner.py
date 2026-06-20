import pandas as pd
import yaml

from src.data.collector import download_ohlcv, resolve_end_date
from src.data.preprocessor import compute_log_returns
from src.features.engineer import (
    compute_rolling_volatility,
    compute_rolling_pairwise_correlations,
)
from src.models.cost_engine import (
    compute_reconstruction_errors,
    compute_trace_scores,
    load_production_model,
)
from src.explainability.shap_explainer import explain_flagged_day, build_business_summary
from src.monitoring.drift_detector import (
    detect_data_drift, detect_corr_drift, detect_model_drift,
)
from src.monitoring.alerting import (
    log_daily_result, send_anomaly_alert, send_drift_alert,
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_fresh_features(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (full feature matrix, log returns) rebuilt from live Yahoo Finance data.

    Reuses collector.download_ohlcv rather than calling yf.download directly — it
    already carries the threads=False fix for the Windows tz-cache race that
    silently drops a ticker's data, and resolve_end_date already handles
    yfinance's exclusive `end` parameter. Reimplementing either here would
    quietly reintroduce bugs Commit 02 already fixed.
    """
    tickers     = config["data"]["tickers"]
    start_date  = config["data"]["start_date"]
    end_date    = resolve_end_date(config["data"]["end_date"])
    corr_window = config["training"]["corr_stability_window"]

    raw   = download_ohlcv(tickers, start_date, end_date)
    close = raw["Close"]
    for ticker in close.columns:
        if close[ticker].isna().all():
            raise ValueError(
                f"Fetch for ticker '{ticker}' returned no data — refusing to build "
                f"today's feature matrix on a poisoned frame. Re-run the pipeline."
            )

    prices  = close.dropna()
    returns = compute_log_returns(prices)

    vol_5d  = compute_rolling_volatility(returns, window=5)
    vol_20d = compute_rolling_volatility(returns, window=20)
    corr    = compute_rolling_pairwise_correlations(returns, window=corr_window)

    return pd.concat([returns, vol_5d, vol_20d, corr], axis=1).dropna(), returns


def normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Applies training normalization to fresh features.

    Using stored mean/std (not recomputed) is critical — recomputing would
    absorb any distribution shift into the normalization, hiding the very
    drift we want to detect.
    """
    mean = pd.read_csv("data/processed/feature_mean.csv", index_col=0).squeeze()
    std  = pd.read_csv("data/processed/feature_std.csv",  index_col=0).squeeze()
    return (features - mean) / std


def run():
    config       = load_config()
    seq_len      = config["data"]["seq_len"]
    tickers      = config["data"]["tickers"]
    alert_config = config["alerting"]

    features, returns   = fetch_fresh_features(config)
    features_normalized = normalize_features(features)

    # The score is "for" the most recent trading day yfinance actually
    # returned, not the wall-clock date — those differ on weekends and
    # holidays. Using date.today() here would mislabel the score and make
    # explain_flagged_day() look up a date that isn't in the trading-day
    # index at all.
    today = features.index[-1].strftime("%Y-%m-%d")
    print(f"Running TRACE for {today}...")

    model = load_production_model()
    model.eval()

    errors      = compute_reconstruction_errors(
        model, features_normalized.values, seq_len, len(tickers)
    )
    error_index = features.index[seq_len - 1:]
    base_errors = pd.DataFrame(errors, index=error_index, columns=tickers)
    trace_scores, portfolio_trace = compute_trace_scores(
        base_errors, returns, config["cost_engine"]
    )
    todays_score = float(portfolio_trace.iloc[-1])

    training_features = pd.read_csv(
        "data/processed/features_normal.csv", index_col=0, parse_dates=True
    )
    historical_trace = pd.read_csv(
        "data/processed/portfolio_trace.csv", index_col=0, parse_dates=True
    ).squeeze()
    is_normal = pd.read_csv(
        "data/processed/normal_mask.csv", index_col=0, parse_dates=True
    )["is_normal"].reindex(returns.index, fill_value=False)

    data_drift = detect_data_drift(training_features, features.tail(30),
                                    alert_config["psi_feature_breach_count"])
    corr_drift = detect_corr_drift(returns, is_normal, returns.tail(30))

    # calibrate model drift against normal-day scores only, not the full
    # unfiltered history — see the "gap" note in Commit 11
    normal_baseline_scores = historical_trace.loc[
        historical_trace.index.intersection(is_normal[is_normal].index)
    ]
    model_drift = detect_model_drift(portfolio_trace, normal_baseline_scores,
                                      alert_config["trace_score_threshold"])

    drift_result = {
        "data_drift":  data_drift,
        "corr_drift":  corr_drift,
        "model_drift": model_drift,
    }

    # SHAP runs only on flagged days because it is expensive
    is_anomaly       = todays_score > alert_config["trace_score_threshold"]
    top_contributors = []
    summary          = "No anomaly detected."

    if is_anomaly:
        shap_values      = explain_flagged_day(today, features_normalized, model,
                                                seq_len=seq_len, n_assets=len(tickers))
        top_contributors = [
            {"feature": idx, "shap_value": float(val)}
            for idx, val in shap_values.head(5).items()
        ]
        summary = build_business_summary(shap_values, today)

    log_daily_result(today, todays_score, top_contributors, drift_result,
                     alert_config["trace_score_threshold"])

    if is_anomaly:
        send_anomaly_alert(today, todays_score, top_contributors, summary, config)

    any_drift = (
        data_drift["drift_detected"]
        or corr_drift["drift_detected"]
        or model_drift["mean_drift_detected"]
        or model_drift["flag_rate_drift_detected"]
    )
    if any_drift:
        send_drift_alert(today, drift_result, config)

    print(f"TRACE score: {todays_score:.4f} | Anomaly: {is_anomaly} | Drift: {any_drift}")


if __name__ == "__main__":
    run()