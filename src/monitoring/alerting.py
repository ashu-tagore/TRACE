import logging
import json
import os
import smtplib
from email.mime.text import MIMEText
from pathlib import Path


def setup_logger(log_path: str = "logs/trace.log") -> logging.Logger:
    """Returns a logger writing JSON to file and plain text to console."""
    Path("logs").mkdir(exist_ok=True)
    logger = logging.getLogger("TRACE")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    file_handler    = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def log_daily_result(date: str, score: float,
                     top_contributors: list, drift_result: dict,
                     threshold: float) -> None:
    """Writes a structured JSON log entry for the day's pipeline run."""
    logger = setup_logger()

    is_anomaly = score > threshold
    any_drift  = (
        drift_result.get("data_drift",  {}).get("drift_detected", False)
        or drift_result.get("corr_drift",  {}).get("drift_detected", False)
        or drift_result.get("model_drift", {}).get("mean_drift_detected", False)
        or drift_result.get("model_drift", {}).get("flag_rate_drift_detected", False)
    )

    log_level = logging.WARNING if (is_anomaly or any_drift) else logging.INFO
    entry = {
        "date":                  date,
        "portfolio_trace_score": round(score, 4),
        "anomaly_flagged":       is_anomaly,
        "drift_detected":        any_drift,
        "top_contributors":      top_contributors,
        "drift_detail":          drift_result,
    }
    logger.log(log_level, json.dumps(entry))


ANOMALY_EMAIL_BODY = """
TRACE Alert — {date}

Portfolio TRACE Score: {score:.3f}  (threshold: {threshold:.3f})

Top Contributing Factors:
{contributors}

Summary:
{summary}

Recommended Action: Review flagged sectors. Check correlation breakdown
between top contributing assets.

--
TRACE | Temporal Reconstruction & Anomaly Cost Engine
"""

DRIFT_EMAIL_BODY = """
TRACE Drift Warning — {date}

{details}

Recommended Action: Review the production-aliased model in MLflow (TRACE-LSTMVAE).
Consider retraining on data through {date}.

--
TRACE | Temporal Reconstruction & Anomaly Cost Engine
"""


def send_email(subject: str, body: str, config: dict) -> None:
    """Sends a plain-text email via SMTP. Credentials from environment variables.

    Skips silently (with a logged warning) if credentials are missing —
    the cost of a missed alert is lower than the cost of crashing the
    daily pipeline.
    """
    smtp_user     = os.environ.get("TRACE_SMTP_USER")
    smtp_password = os.environ.get("TRACE_SMTP_PASSWORD")

    if not smtp_user or not smtp_password:
        logging.getLogger("TRACE").warning(
            "Email skipped: TRACE_SMTP_USER or TRACE_SMTP_PASSWORD not set"
        )
        return

    msg            = MIMEText(body)
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = config["alerting"]["email_recipient"]

    with smtplib.SMTP(config["alerting"]["smtp_host"],
                      config["alerting"]["smtp_port"]) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)


def send_anomaly_alert(date: str, score: float, top_contributors: list,
                       summary: str, config: dict) -> None:
    """Sends the anomaly alert email."""
    # shap_value is a SHAP magnitude (np.abs of the raw value), never negative —
    # no sign to display, so this is a plain fixed-point format, not `:+.3f`
    contributors_text = "\n".join(
        f"  {c['feature']}: {c['shap_value']:.3f}"
        for c in top_contributors
    )
    body = ANOMALY_EMAIL_BODY.format(
        date=date, score=score,
        threshold=config["alerting"]["trace_score_threshold"],
        contributors=contributors_text, summary=summary,
    )
    send_email(f"[TRACE] Anomaly Alert — {date}", body, config)


def build_drift_details(drift_result: dict) -> str:
    """Returns one human-readable line per drift check that actually fired.

    drift_result is the same {"data_drift", "corr_drift", "model_drift"} dict
    logged by log_daily_result — the three checks are independent, so the email
    has to be able to describe any combination of them, not just model drift.
    """
    lines = []

    data_drift = drift_result.get("data_drift", {})
    if data_drift.get("drift_detected"):
        lines.append(
            f"Data drift (PSI): {data_drift.get('n_features_drifted', 0)} features "
            f"breached PSI > 0.25 (max PSI: {data_drift.get('max_psi', 0.0):.3f})."
        )

    corr_drift = drift_result.get("corr_drift", {})
    if corr_drift.get("drift_detected"):
        lines.append(
            f"Correlation drift: distance {corr_drift.get('corr_distance', 0.0):.3f} "
            f"exceeds threshold {corr_drift.get('threshold', 0.0):.3f}."
        )

    model_drift = drift_result.get("model_drift", {})
    if model_drift.get("mean_drift_detected"):
        lines.append(
            f"Model drift (mean inflation): recent avg score "
            f"{model_drift.get('recent_mean', 0.0):.4f} vs baseline "
            f"{model_drift.get('baseline_mean', 0.0):.4f}."
        )
    if model_drift.get("flag_rate_drift_detected"):
        lines.append(
            f"Model drift (flag rate): recent flag rate "
            f"{model_drift.get('recent_flag_rate', 0.0):.1%} vs baseline "
            f"{model_drift.get('baseline_flag_rate', 0.0):.1%}."
        )

    return "\n".join(lines)


def send_drift_alert(date: str, drift_result: dict, config: dict) -> None:
    """Sends the drift warning email, describing whichever check(s) fired."""
    body = DRIFT_EMAIL_BODY.format(date=date, details=build_drift_details(drift_result))
    send_email(f"[TRACE] Drift Warning — {date}", body, config)