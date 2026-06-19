import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_normal_sequences(features: np.ndarray, is_normal: np.ndarray,
                           seq_len: int, stride: int) -> np.ndarray:
    """Returns sliding-window sequences drawn only from runs of consecutive normal days.

    Windowing over calendar position in the full feature matrix — rather than over
    a normal-only subset — keeps every window a genuine run of consecutive trading
    days. A window is kept only if every day inside it is normal; one that touches
    even a single excluded day is dropped instead of being silently spliced to
    whatever normal day comes next, which would teach the model that a multi-month
    jump (e.g. across the COVID exclusion window) is a normal 30-day trajectory.
    Shape: [n_sequences, seq_len, n_features].
    """
    n_days  = len(features)
    starts  = range(0, n_days - seq_len + 1, stride)
    windows = [
        features[s : s + seq_len]
        for s in starts
        if is_normal[s : s + seq_len].all()
    ]
    return np.array(windows)


def build():
    config    = load_config()
    seq_len   = config["data"]["seq_len"]
    stride    = config["data"]["stride"]
    val_split = config["training_loop"]["val_split"]

    features = pd.read_csv(
        "data/processed/features.csv", index_col=0, parse_dates=True
    )
    normal_mask = pd.read_csv(
        "data/processed/normal_mask.csv", index_col=0, parse_dates=True
    )["is_normal"].reindex(features.index, fill_value=False)

    # split chronologically before computing normalization stats —
    # fitting on the full dataset would leak val statistics into train
    split_idx      = int(len(features) * (1 - val_split))
    train_features = features.iloc[:split_idx]
    train_mask     = normal_mask.iloc[:split_idx]
    val_features   = features.iloc[split_idx:]
    val_mask       = normal_mask.iloc[split_idx:]

    # fit normalization only on normal days within the training portion —
    # matches the data the model is actually trained on
    mean = train_features[train_mask].mean()
    std  = train_features[train_mask].std().replace(0, 1.0)   # avoid /0 on constant columns

    train_normalized = (train_features - mean) / std
    val_normalized   = (val_features   - mean) / std

    output_dir = Path("data/processed")
    mean.to_csv(output_dir / "feature_mean.csv", header=["mean"])
    std.to_csv(output_dir  / "feature_std.csv",  header=["std"])

    # building sequences after the split guarantees no sequence spans
    # the train/val boundary — another path through which val data could leak
    train_seqs = build_normal_sequences(
        train_normalized.values.astype(np.float32), train_mask.values, seq_len, stride
    )
    val_seqs = build_normal_sequences(
        val_normalized.values.astype(np.float32), val_mask.values, seq_len, stride
    )

    np.save(output_dir / "train_sequences.npy", train_seqs)
    np.save(output_dir / "val_sequences.npy",   val_seqs)

    print(f"Train: {train_seqs.shape}  (days 0..{split_idx})")
    print(f"Val:   {val_seqs.shape}    (days {split_idx}..{len(features)})")


if __name__ == "__main__":
    build()