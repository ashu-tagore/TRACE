import pandas as pd
import numpy as np
import shap
import torch
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def explain_flagged_day(flagged_date: str,
                        features_normalized: pd.DataFrame,
                        model,
                        seq_len: int,
                        n_assets: int) -> pd.Series:
    """Returns feature importance scores for a flagged day, sorted descending.

    The black-box function passed to KernelSHAP must compute the *same*
    quantity that triggered the alert. The cost engine reads
    reconstruction error from `recon[:, -1, :n_assets]` — last timestep,
    log-return features only. Computing SHAP against the full-window
    MSE would explain a different metric than the one that actually
    flagged the day.
    """
    feature_names = features_normalized.columns.tolist()
    n_features    = len(feature_names)
    device        = next(model.parameters()).device
    date_idx      = features_normalized.index.get_loc(flagged_date)

    if date_idx < seq_len:
        raise ValueError(f"Not enough history before {flagged_date} to build a {seq_len}-day window")

    flagged_window = features_normalized.values[date_idx - seq_len : date_idx]
    flagged_flat   = flagged_window.flatten().reshape(1, -1)

    # 50 evenly-spaced historical windows serve as the baseline against
    # which feature impact is measured — SHAP values are always relative
    step = max(1, (len(features_normalized) - seq_len) // 50)
    background_flat = np.array([
        features_normalized.values[i : i + seq_len].flatten()
        for i in range(0, len(features_normalized) - seq_len, step)
    ])[:50]

    def predict_reconstruction_error(windows_flat: np.ndarray) -> np.ndarray:
        n_samples  = len(windows_flat)
        windows_3d = windows_flat.reshape(n_samples, seq_len, n_features)
        errors     = []
        model.eval()
        with torch.no_grad():
            for w in windows_3d:
                x          = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)
                recon, _, _ = model(x)
                # match the cost engine's target exactly: last timestep,
                # first n_assets columns (log returns)
                last_true   = x[:, -1, :n_assets]
                last_recon  = recon[:, -1, :n_assets]
                mse         = ((last_true - last_recon) ** 2).mean().item()
                errors.append(mse)
        return np.array(errors)

    explainer   = shap.KernelExplainer(predict_reconstruction_error, background_flat)
    shap_values = explainer.shap_values(flagged_flat, nsamples=100)

    per_timestep = shap_values[0].reshape(seq_len, n_features)
    per_feature  = np.abs(per_timestep).mean(axis=0)

    return pd.Series(per_feature, index=feature_names).sort_values(ascending=False)


def build_business_summary(top_shap: pd.Series, date: str) -> str:
    """Returns a plain-language explanation for a portfolio manager."""
    top_feature    = top_shap.index[0]
    second_feature = top_shap.index[1] if len(top_shap) > 1 else "secondary factor"
    primary_asset  = top_feature.split("_")[0]

    return (
        f"TRACE flagged {date}: primarily driven by {primary_asset} "
        f"({top_feature}, {second_feature}). "
        f"Review {primary_asset} sector exposure and check for correlation breakdown."
    )